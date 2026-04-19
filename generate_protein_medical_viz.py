"""
generate_protein_medical_viz.py  (v2 -- PyVista/VTK renderer)
==============================================================
Generates professional-quality protein structure images using:

  PyVista (VTK 9)  -- same rendering engine as ParaView, VisIt, UCSF ChimeraX
  scikit-image     -- marching cubes for VDW molecular surface
  matplotlib       -- thin connector / 2-D topology only

Why PyVista not matplotlib:
  matplotlib 3D = sketchy lines, no real lighting, Z-sort artifacts.
  PyVista = GPU-quality ribbon shading, ambient occlusion, anti-aliasing,
            proper perspective, depth-of-field -- matches PyMOL/Chimera look.

Outputs:
  figures/43a_docked_complex_ESR1.png     -- rainbow ribbon + tamoxifen
  figures/43b_molecular_surface_ESR1.png  -- VDW ESP surface
  figures/43c_protein_panels_ESR1.png     -- (a)+(b) side by side
  (same for ERBB2 and PIK3CA)
  figures/43_combined_docked_surface.png  -- all 3 proteins combined

Run:   python generate_protein_medical_viz.py
"""

import os, warnings, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
np.random.seed(42)

import pyvista as pv
pv.global_theme.allow_empty_mesh = True

ROOT    = Path("d:/Aakanksha/thesis/onco-fusion")
PDB_DIR = ROOT / "data" / "pdb_cache"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGETS = [
    #  pdb_file    name      ligands     subtype  chain  view_vec              pocket_r
    ("3ERT.pdb",  "ESR1",   ["OHT"],    "HR+",   "A",  [0.55, 0.40,  1.0 ],  999),
    # 2JIV: ERBB2 kinase domain + HKI (lapatinib analog), ligand 0.71 A from protein
    ("2JIV.pdb",  "ERBB2",  ["HKI"],    "HER2+", "A",  [0.55, 0.40,  1.0 ],  999),
    # PIK3CA (1005 res): show only binding-pocket region (~28 A) to avoid messy full-chain view
    ("4JPS.pdb",  "PIK3CA", ["1LT","1W6"],"TNBC", "A",  [0.55, 0.40,  1.0 ],  28),
]

ELEM_CPK = {
    "C":"#888888","N":"#3366DD","O":"#DD3333","S":"#DDBB00",
    "F":"#33CC33","CL":"#33CC33","BR":"#993300","P":"#FF8800","H":"#EEEEEE",
}
ELEM_RADIUS = {"C":0.77,"N":0.75,"O":0.73,"S":1.03,"P":1.06,"H":0.37}

POS_RES = {"ARG","LYS","HIS"}
NEG_RES = {"ASP","GLU"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDB PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb(pdb_path, chain_id="A"):
    ca_atoms, all_atoms, hetatms = [], [], []
    helices, sheets = [], []
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "HELIX":
                try:
                    c = line[19]; s = int(line[21:25]); e = int(line[33:37])
                    helices.append((c, s, e))
                except: pass
            elif rec == "SHEET":
                try:
                    c = line[21]; s = int(line[22:26]); e = int(line[33:37])
                    sheets.append((c, s, e))
                except: pass
            elif rec == "ATOM":
                try:
                    an = line[12:16].strip(); rn = line[17:20].strip()
                    ch = line[21]; rnum = int(line[22:26])
                    x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
                    el = line[76:78].strip() if len(line)>77 else an[:1]
                    all_atoms.append((ch,rnum,rn,an,el,x,y,z))
                    if an == "CA" and ch == chain_id:
                        ca_atoms.append((rnum,x,y,z))
                except: pass
            elif rec == "HETATM":
                try:
                    rn = line[17:20].strip()
                    if rn in ("HOH","WAT","DOD"): continue
                    an = line[12:16].strip(); rnum = int(line[22:26])
                    x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
                    el = line[76:78].strip() if len(line)>77 else an[:1]
                    hetatms.append((rnum,rn,an,el,x,y,z))
                except: pass
    ss_helix, ss_sheet = set(), set()
    for c,s,e in helices:
        if c==chain_id: ss_helix.update(range(s,e+1))
    for c,s,e in sheets:
        if c==chain_id: ss_sheet.update(range(s,e+1))
    ca_atoms.sort(key=lambda a:a[0])
    return ca_atoms, all_atoms, hetatms, ss_helix, ss_sheet


def filter_ligand(hetatms, resnames):
    for rn in resnames:
        atoms = [(r,res,nm,el,x,y,z) for r,res,nm,el,x,y,z in hetatms if res==rn]
        if len(atoms) >= 3:
            return atoms, rn
    if hetatms:
        from collections import Counter
        cnt = Counter(res for _,res,*_ in hetatms)
        best = cnt.most_common(1)[0][0]
        return [(r,res,nm,el,x,y,z) for r,res,nm,el,x,y,z in hetatms if res==best], best
    return [], ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. SMOOTH SPLINE
# ─────────────────────────────────────────────────────────────────────────────

def smooth_spline(pts, n_seg=15):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 4:
        return pts
    try:
        tck, u = splprep([pts[:,0],pts[:,1],pts[:,2]], s=1.5, k=3)
        u_f = np.linspace(0,1, len(pts)*n_seg)
        return np.column_stack(splev(u_f, tck))
    except:
        return pts


def get_segments(resnums, ss_helix, ss_sheet):
    """Return list of (ss_type, start_idx, end_idx) with >=2 residues."""
    segs, i = [], 0
    n = len(resnums)
    while i < n:
        r = resnums[i]
        ss = "H" if r in ss_helix else "E" if r in ss_sheet else "L"
        j = i+1
        while j < n:
            rj = resnums[j]
            ssj = "H" if rj in ss_helix else "E" if rj in ss_sheet else "L"
            if ssj != ss: break
            j += 1
        if j - i >= 2:
            segs.append((ss, i, j))
        i = j
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# 3. PYVISTA RIBBON RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def jet_color(t):
    """Return (R,G,B) for t in [0,1] using jet colormap."""
    import matplotlib.cm as cm
    r,g,b,_ = cm.jet(float(t))
    return (r,g,b)


def assign_rainbow(mesh, t_start, t_end):
    """Add 'rainbow_t' scalars uniformly across mesh points."""
    n = mesh.n_points
    mesh["rainbow_t"] = np.linspace(t_start, t_end, n)
    return mesh


def build_ribbon_plotter(ca_atoms, hetatm_lig, ss_helix, ss_sheet,
                          window_size=(1100, 900),
                          view_vec=None,
                          pocket_radius=30.0):
    """
    Render rainbow ribbon + ligand using PyVista/VTK (off-screen).
    For large proteins (>300 residues) only residues within pocket_radius A
    of the ligand centre are rendered -- keeps the view clean and relevant.
    Returns pv.Plotter (call .screenshot() to save PNG).
    """
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background("white")

    resnums = [r for r,x,y,z in ca_atoms]
    coords  = np.array([[x,y,z] for r,x,y,z in ca_atoms])
    n_res   = len(coords)
    if n_res < 3:
        return pl

    # For large proteins, restrict to binding-pocket neighbourhood
    if n_res > 300 and hetatm_lig:
        lig_c = np.array([[x,y,z] for _,_,_,el,x,y,z in hetatm_lig]).mean(0)
        dists = np.linalg.norm(coords - lig_c, axis=1)
        keep  = dists < pocket_radius
        ca_atoms  = [a for a, k in zip(ca_atoms, keep) if k]
        coords    = coords[keep]
        resnums   = [r for r, k in zip(resnums, keep) if k]
        n_res     = len(coords)
        if n_res < 3:
            return pl

    segments = get_segments(resnums, ss_helix, ss_sheet)

    added = 0
    for ss, s_idx, e_idx in segments:
        seg_pts = coords[s_idx : e_idx+1]
        if len(seg_pts) < 2:
            continue

        t_s = s_idx / max(n_res-1,1)
        t_e = min(e_idx, n_res-1) / max(n_res-1,1)

        smooth = smooth_spline(seg_pts, n_seg=12)
        if len(smooth) < 2:
            continue

        try:
            poly = pv.Spline(smooth, n_points=max(len(smooth), 40))
        except Exception:
            poly = pv.PolyData(smooth)
            continue

        try:
            if ss == "H":
                # Alpha-helix: wide smooth tube
                mesh = poly.tube(radius=0.75, n_sides=16)
                mesh = assign_rainbow(mesh, t_s, t_e)
                pl.add_mesh(mesh, scalars="rainbow_t", cmap="jet",
                            clim=[0,1], smooth_shading=True,
                            show_scalar_bar=False, lighting=True,
                            ambient=0.3, diffuse=0.7, specular=0.2,
                            specular_power=20)

            elif ss == "E":
                # Beta-sheet: flat ribbon
                # Build normal perpendicular to tangent
                tangent = smooth[-1] - smooth[0]
                up = np.array([0,1,0])
                if abs(np.dot(tangent/np.linalg.norm(tangent+1e-9), up)) > 0.85:
                    up = np.array([1,0,0])
                try:
                    mesh = poly.ribbon(width=1.1, normal=up, tcoords=True)
                    mesh = assign_rainbow(mesh, t_s, t_e)
                    pl.add_mesh(mesh, scalars="rainbow_t", cmap="jet",
                                clim=[0,1], smooth_shading=True,
                                show_scalar_bar=False, lighting=True,
                                ambient=0.3, diffuse=0.8, specular=0.15)
                    # Arrow tip: cone at C-end
                    tip_pts  = smooth[-8:] if len(smooth)>=8 else smooth
                    tip_dir  = tip_pts[-1] - tip_pts[0]
                    tip_len  = np.linalg.norm(tip_dir)
                    if tip_len > 0.1:
                        tip_dir /= tip_len
                        cone = pv.Cone(center=tip_pts[-1],
                                       direction=tip_dir,
                                       height=1.4, radius=0.7)
                        r,g,b = jet_color(t_e)
                        pl.add_mesh(cone, color=(r,g,b), smooth_shading=True,
                                    lighting=True, ambient=0.3, diffuse=0.8)
                except Exception:
                    mesh2 = poly.tube(radius=0.35, n_sides=6)
                    mesh2 = assign_rainbow(mesh2, t_s, t_e)
                    pl.add_mesh(mesh2, scalars="rainbow_t", cmap="jet",
                                clim=[0,1], smooth_shading=True,
                                show_scalar_bar=False, lighting=True)

            else:
                # Loop: thin tube
                mesh = poly.tube(radius=0.22, n_sides=8)
                mesh = assign_rainbow(mesh, t_s, t_e)
                pl.add_mesh(mesh, scalars="rainbow_t", cmap="jet",
                            clim=[0,1], smooth_shading=True,
                            show_scalar_bar=False, lighting=True,
                            ambient=0.2, diffuse=0.8)

            added += 1
        except Exception as exc:
            continue

    # ── Ligand: spheres + stick bonds ───────────────────────────────────────
    if hetatm_lig:
        lig_pts = np.array([[x,y,z] for _,_,_,el,x,y,z in hetatm_lig])
        lig_el  = [el.upper().strip() if el.strip() else "C"
                   for _,_,_,el,x,y,z in hetatm_lig]

        from scipy.spatial.distance import cdist
        dists_lig = cdist(lig_pts, lig_pts)

        for i, (pt, el) in enumerate(zip(lig_pts, lig_el)):
            col = ELEM_CPK.get(el, "#888888")
            rad = ELEM_RADIUS.get(el, 0.77) * 0.55
            sph = pv.Sphere(radius=rad, center=pt, theta_resolution=16,
                            phi_resolution=16)
            pl.add_mesh(sph, color=col, smooth_shading=True,
                        ambient=0.25, diffuse=0.75, specular=0.4,
                        specular_power=30)

        n_lig = len(lig_pts)
        for i in range(n_lig):
            for j in range(i+1, n_lig):
                if 0.5 < dists_lig[i,j] < 2.0:
                    mid = (lig_pts[i] + lig_pts[j]) / 2
                    c1 = ELEM_CPK.get(lig_el[i], "#888888")
                    c2 = ELEM_CPK.get(lig_el[j], "#888888")
                    for seg_pts_bond, col_bond in [
                            (np.array([lig_pts[i], mid]), c1),
                            (np.array([mid, lig_pts[j]]), c2)]:
                        line = pv.Spline(seg_pts_bond, n_points=3)
                        cyl  = line.tube(radius=0.12, n_sides=8)
                        pl.add_mesh(cyl, color=col_bond, smooth_shading=True,
                                    ambient=0.2, diffuse=0.8, specular=0.3)

    # ── Camera: use reset_camera() to auto-fit ALL geometry, then fix
    #   the km-scale clipping range bug that reset_camera() introduces.
    #   The view_vec parameter is applied via a pre-set position BEFORE
    #   reset_camera so the orbital angle is respected.
    prot_center = coords.mean(axis=0)
    prot_span   = (coords.max(axis=0) - coords.min(axis=0)).max()

    if view_vec is None:
        view_vec = np.array([0.55, 0.40, 1.0], dtype=float)
    view_vec = np.asarray(view_vec, dtype=float)
    view_vec /= np.linalg.norm(view_vec)

    # Set a reasonable starting position (reset_camera will adjust distance)
    pl.camera.position    = (prot_center + view_vec * prot_span * 3.0).tolist()
    pl.camera.focal_point = prot_center.tolist()
    pl.camera.up          = [0, 1, 0]

    # reset_camera moves camera to exactly fit all geometry in view
    pl.reset_camera()

    # Zoom out 15% to add white-space margin around the protein
    pl.camera.zoom(1.30)

    # reset_camera sets a km-scale clipping range -- override it
    pl.camera.clipping_range = (0.1, 5000.0)

    # Lighting
    pl.add_light(pv.Light(position=(50, 50, 100), light_type="headlight",
                          intensity=0.7))

    return pl


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYVISTA MOLECULAR SURFACE RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def build_surface_plotter(all_atoms, chain_id, hetatm_lig,
                           window_size=(1100, 900),
                           pocket_only=True, pocket_radius=20.0,
                           view_vec=None):
    """
    Render VDW molecular surface coloured by ESP using PyVista/VTK.
    Returns pv.Plotter (call .screenshot() to save PNG).
    """
    from skimage import measure
    import matplotlib.colors as mcolors

    CHARGE = {"ARG":+1,"LYS":+1,"HIS":+0.5,"ASP":-1,"GLU":-1}

    # Protein heavy atoms (chain_id only)
    pts, charges, resnames_list = [], [], []
    for c,rnum,rn,an,el,x,y,z in all_atoms:
        if c != chain_id: continue
        if an.startswith("H"): continue
        pts.append([x,y,z])
        charges.append(CHARGE.get(rn, 0.0))
        resnames_list.append(rn)

    if not pts:
        return pv.Plotter(off_screen=True)

    pts     = np.array(pts, dtype=float)
    charges = np.array(charges, dtype=float)

    # If pocket_only: restrict to atoms near ligand centre
    if pocket_only and hetatm_lig:
        lig_pts = np.array([[x,y,z] for _,_,_,el,x,y,z in hetatm_lig])
        lig_c   = lig_pts.mean(axis=0)
        d       = np.linalg.norm(pts - lig_c, axis=1)
        mask    = d < pocket_radius
        pts     = pts[mask]
        charges = charges[mask]

    if len(pts) < 10:
        return pv.Plotter(off_screen=True)

    # Grid
    grid_step = 1.1
    mn = pts.min(axis=0) - 4
    mx = pts.max(axis=0) + 4
    nx = max(18, int((mx[0]-mn[0])/grid_step))
    ny = max(18, int((mx[1]-mn[1])/grid_step))
    nz = max(18, int((mx[2]-mn[2])/grid_step))
    while nx*ny*nz > 400_000:
        grid_step *= 1.12
        nx = max(18, int((mx[0]-mn[0])/grid_step))
        ny = max(18, int((mx[1]-mn[1])/grid_step))
        nz = max(18, int((mx[2]-mn[2])/grid_step))

    density  = np.zeros((nx,ny,nz), dtype=np.float32)
    esp_grid = np.zeros((nx,ny,nz), dtype=np.float32)

    xi = np.linspace(mn[0], mx[0], nx)
    yi = np.linspace(mn[1], mx[1], ny)
    zi = np.linspace(mn[2], mx[2], nz)

    for (x,y,z), chg in zip(pts, charges):
        ix = int(np.clip((x-mn[0])/(mx[0]-mn[0]+1e-9)*(nx-1), 0, nx-1))
        iy = int(np.clip((y-mn[1])/(mx[1]-mn[1]+1e-9)*(ny-1), 0, ny-1))
        iz = int(np.clip((z-mn[2])/(mx[2]-mn[2]+1e-9)*(nz-1), 0, nz-1))
        density[ix,iy,iz] += 1.0
        esp_grid[ix,iy,iz] += chg

    density  = gaussian_filter(density,  sigma=1.8/grid_step)
    esp_grid = gaussian_filter(esp_grid, sigma=2.5/grid_step)

    level = density.max() * 0.07
    if level < 1e-7:
        return pv.Plotter(off_screen=True)

    verts, faces, normals, _ = measure.marching_cubes(density, level=level)

    # Map verts back to Angstrom
    v_A = np.column_stack([
        xi[np.clip(verts[:,0].astype(int), 0, nx-1)],
        yi[np.clip(verts[:,1].astype(int), 0, ny-1)],
        zi[np.clip(verts[:,2].astype(int), 0, nz-1)],
    ])

    # Interpolate ESP at each vertex
    vix = np.clip(verts[:,0].astype(int),0,nx-1)
    viy = np.clip(verts[:,1].astype(int),0,ny-1)
    viz = np.clip(verts[:,2].astype(int),0,nz-1)
    esp_v = esp_grid[vix, viy, viz]
    vmax  = max(np.abs(esp_v).max(), 0.05)
    esp_n = np.clip(esp_v / vmax, -1, 1)

    # Build PyVista mesh
    faces_pv = np.column_stack([np.full(len(faces),3), faces]).astype(np.int_)
    surf = pv.PolyData(v_A, faces_pv.flatten())
    surf["esp"] = esp_n
    surf = surf.smooth(n_iter=50)           # smooth the surface
    surf.compute_normals(inplace=True)

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background("white")

    # Blue=positive, White=neutral, Red=negative  (same as PyMOL ESP)
    cmap_esp = mcolors.LinearSegmentedColormap.from_list(
        "esp", ["#CC2222", "#FFFFFF", "#2222CC"])

    # Semi-transparent surface so the ligand inside is visible
    pl.add_mesh(surf, scalars="esp", cmap=cmap_esp,
                clim=[-1, 1], opacity=0.52,
                smooth_shading=True, show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Electrostatic\nPotential",
                    "title_font_size": 13,
                    "label_font_size": 11,
                    "n_labels": 3,
                    "color": "black",
                    "position_x": 0.78,
                    "position_y": 0.25,
                })

    # Ligand: full ball-and-stick (same as ribbon plotter) so it shows through surface
    if hetatm_lig:
        lig_pts  = np.array([[x,y,z] for _,_,_,el,x,y,z in hetatm_lig])
        lig_el   = [el.upper().strip() if el.strip() else "C"
                    for _,_,_,el,x,y,z in hetatm_lig]

        from scipy.spatial.distance import cdist
        dists_lig = cdist(lig_pts, lig_pts)

        for pt, el in zip(lig_pts, lig_el):
            col = ELEM_CPK.get(el, "#888888")
            rad = ELEM_RADIUS.get(el, 0.77) * 0.60
            sph = pv.Sphere(radius=rad, center=pt,
                            theta_resolution=16, phi_resolution=16)
            pl.add_mesh(sph, color=col, smooth_shading=True,
                        ambient=0.25, diffuse=0.75, specular=0.5,
                        specular_power=40)

        n_lig = len(lig_pts)
        for i in range(n_lig):
            for j in range(i+1, n_lig):
                if 0.5 < dists_lig[i,j] < 2.0:
                    mid = (lig_pts[i] + lig_pts[j]) / 2
                    c1  = ELEM_CPK.get(lig_el[i], "#888888")
                    c2  = ELEM_CPK.get(lig_el[j], "#888888")
                    for seg_pts_bond, col_bond in [
                            (np.array([lig_pts[i], mid]), c1),
                            (np.array([mid, lig_pts[j]]), c2)]:
                        line = pv.Spline(seg_pts_bond, n_points=3)
                        cyl  = line.tube(radius=0.13, n_sides=8)
                        pl.add_mesh(cyl, color=col_bond, smooth_shading=True,
                                    ambient=0.2, diffuse=0.8, specular=0.3)
        center = lig_pts.mean(axis=0)
    else:
        center = v_A.mean(axis=0)

    surf_center = v_A.mean(axis=0)
    surf_span   = (v_A.max(axis=0) - v_A.min(axis=0)).max()
    if view_vec is None:
        view_vec = np.array([0.55, 0.40, 1.0], dtype=float)
    view_vec = np.asarray(view_vec, dtype=float)
    view_vec /= np.linalg.norm(view_vec)

    # Same reset_camera approach: set viewing angle, auto-fit, fix clipping
    pl.camera.position    = (surf_center + view_vec * surf_span * 3.0).tolist()
    pl.camera.focal_point = surf_center.tolist()
    pl.camera.up          = [0, 1, 0]
    pl.reset_camera()
    pl.camera.zoom(1.30)
    pl.camera.clipping_range = (0.1, 5000.0)
    pl.add_light(pv.Light(position=(50, 50, 100), light_type="headlight",
                          intensity=0.7))

    return pl


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE PER-PROTEIN IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def make_protein_images(pdb_path, protein_name, lig_resnames, chain_id, subtype,
                         view_vec=None, pocket_radius=999):
    print(f"\n[{protein_name}]  ({subtype})")
    ca_atoms, all_atoms, hetatms, ss_helix, ss_sheet = parse_pdb(
        pdb_path, chain_id)

    if not ca_atoms:
        print(f"  No CA atoms for chain {chain_id} -- skip")
        return

    hetatm_lig, lig_name = filter_ligand(hetatms, lig_resnames)
    print(f"  Parsing: {len(ca_atoms)} residues | "
          f"{len(ss_helix)} helix | {len(ss_sheet)} sheet | "
          f"ligand {lig_name} ({len(hetatm_lig)} atoms)")

    # ── Panel (a): Ribbon + docked ligand ─────────────────────────────────
    print("  Rendering ribbon (PyVista/VTK) ...")
    try:
        pl_a = build_ribbon_plotter(ca_atoms, hetatm_lig, ss_helix, ss_sheet,
                                     window_size=(1100, 900),
                                     view_vec=view_vec,
                                     pocket_radius=pocket_radius)
        pl_a.enable_anti_aliasing("ssaa")
        out_a = FIG_DIR / f"43a_docked_complex_{protein_name}.png"
        pl_a.screenshot(str(out_a), transparent_background=False)
        pl_a.close()
        print(f"  Saved: {out_a.name}  ({out_a.stat().st_size//1024} KB)")
    except Exception as exc:
        import traceback; traceback.print_exc()
        out_a = None

    gc.collect()

    # ── Panel (b): Molecular surface ──────────────────────────────────────
    print("  Computing molecular surface (marching cubes) ...")
    try:
        pl_b = build_surface_plotter(all_atoms, chain_id, hetatm_lig,
                                      window_size=(1100, 900),
                                      pocket_only=True, pocket_radius=22.0,
                                      view_vec=view_vec)
        pl_b.enable_anti_aliasing("ssaa")
        out_b = FIG_DIR / f"43b_molecular_surface_{protein_name}.png"
        pl_b.screenshot(str(out_b), transparent_background=False)
        pl_b.close()
        print(f"  Saved: {out_b.name}  ({out_b.stat().st_size//1024} KB)")
    except Exception as exc:
        import traceback; traceback.print_exc()
        out_b = None

    gc.collect()

    # ── Combined panel (a)+(b) ─────────────────────────────────────────────
    if out_a and out_b and out_a.exists() and out_b.exists():
        try:
            from PIL import Image as PILImage
            img_a = PILImage.open(out_a).convert("RGB")
            img_b = PILImage.open(out_b).convert("RGB")
            h     = min(img_a.height, img_b.height)
            w     = min(img_a.width,  img_b.width)
            img_a = img_a.resize((w,h), PILImage.LANCZOS)
            img_b = img_b.resize((w,h), PILImage.LANCZOS)

            gap    = 12
            canvas = PILImage.new("RGB", (2*w+gap, h+60), (255,255,255))
            canvas.paste(img_a, (0, 60))
            canvas.paste(img_b, (w+gap, 60))

            # Header text
            import matplotlib.pyplot as plt2
            import matplotlib.patches as mpatches
            fig_hdr, ax_hdr = plt2.subplots(figsize=((2*w+gap)/100, 60/100),
                                             facecolor="white")
            ax_hdr.text(0.25, 0.5, "(a) Docked complex",
                        transform=ax_hdr.transAxes, ha="center", va="center",
                        fontsize=18, fontweight="bold", color="#222222",
                        fontfamily="DejaVu Sans")
            ax_hdr.text(0.75, 0.5, "(b) molecular surface",
                        transform=ax_hdr.transAxes, ha="center", va="center",
                        fontsize=18, fontweight="bold", color="#222222",
                        fontfamily="DejaVu Sans")
            ax_hdr.axis("off")
            import tempfile
            tmp_hdr = tempfile.mktemp(suffix=".png")
            fig_hdr.savefig(tmp_hdr, dpi=100, bbox_inches="tight",
                            facecolor="white")
            plt2.close(fig_hdr)
            hdr = PILImage.open(tmp_hdr).convert("RGB").resize((2*w+gap, 60))
            canvas.paste(hdr, (0, 0))
            import os; os.unlink(tmp_hdr)

            out_c = FIG_DIR / f"43c_protein_panels_{protein_name}.png"
            canvas.save(out_c, dpi=(150,150))
            print(f"  Saved: {out_c.name}  ({out_c.stat().st_size//1024} KB)")
        except Exception as exc:
            print(f"  Combined panel failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMBINED 3-PROTEIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

def make_combined_panel():
    from PIL import Image as PILImage
    proteins = ["ESR1","ERBB2","PIK3CA"]
    panels = []
    for prot in proteins:
        p = FIG_DIR / f"43c_protein_panels_{prot}.png"
        if p.exists():
            panels.append(PILImage.open(p).convert("RGB"))
    if not panels:
        return
    h = min(im.height for im in panels)
    w = min(im.width  for im in panels)
    panels = [im.resize((w,h), PILImage.LANCZOS) for im in panels]
    gap = 8
    canvas = PILImage.new("RGB", (len(panels)*w+(len(panels)-1)*gap, h),
                          (255,255,255))
    for i,im in enumerate(panels):
        canvas.paste(im, (i*(w+gap), 0))
    out = FIG_DIR / "43_combined_docked_surface.png"
    canvas.save(out)
    print(f"\nSaved: {out.name}  ({out.stat().st_size//1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 7. UPDATE 11_drug_discovery.ipynb WITH py3Dmol CELLS
# ─────────────────────────────────────────────────────────────────────────────

PY3DMOL_CELL = '''\
# =========================================================================
# INTERACTIVE 3D VIEWER -- {prot}  ({subtype})
# Uses py3Dmol (WebGL renderer) -- same quality as PyMOL/UCSF Chimera
# (a) Rainbow cartoon ribbon + docked ligand sticks/spheres
# (b) Van der Waals surface coloured by electrostatic potential
# =========================================================================
import py3Dmol, IPython.display as ipd
from pathlib import Path

pdb_data = open(r"d:/Aakanksha/thesis/onco-fusion/data/pdb_cache/{pdb}").read()

# ---- (a) Docked complex --------------------------------------------------
v1 = py3Dmol.view(width=560, height=450, linked=False)
v1.addModel(pdb_data, "pdb")
v1.setStyle({{"chain":"{chain}"}},
            {{"cartoon":{{"color":"spectrum","thickness":0.4,"opacity":1.0}}}})
v1.addStyle({{"hetflag":True,"not":{{"resn":"HOH"}}}},
            {{"stick":{{"colorscheme":"grayCarbon","radius":0.18}}}})
v1.addStyle({{"hetflag":True,"not":{{"resn":"HOH"}}}},
            {{"sphere":{{"colorscheme":"grayCarbon","scale":0.30}}}})
v1.setBackgroundColor("white")
v1.zoomTo({{"hetflag":True,"not":{{"resn":"HOH"}}}})
v1.zoom(0.85)
print("(a) {prot} -- rainbow ribbon cartoon + {lig_hint}")
ipd.display(v1.show())

# ---- (b) Molecular surface (ESP) ----------------------------------------
v2 = py3Dmol.view(width=560, height=450, linked=False)
v2.addModel(pdb_data, "pdb")
v2.setStyle({{"chain":"{chain}"}},
            {{"cartoon":{{"color":"spectrum","opacity":0.12}}}})
v2.addSurface(
    py3Dmol.VDW,
    {{"opacity":0.90,
      "colorscheme":{{"prop":"partialCharge","gradient":"rwb",
                     "min":-0.6,"max":0.6}}}},
    {{"chain":"{chain}"}}
)
v2.setBackgroundColor("white")
v2.zoomTo({{"chain":"{chain}"}})
print("(b) {prot} -- molecular surface (blue=+, red=-, white=neutral)")
ipd.display(v2.show())
'''

def update_notebook():
    """Re-inject clean py3Dmol cells into 11_drug_discovery.ipynb."""
    import nbformat
    nb_path = ROOT / "notebooks" / "11_drug_discovery.ipynb"
    if not nb_path.exists():
        print("Notebook not found -- skipping")
        return
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Remove ALL previously injected cells -- catch every marker pattern from past runs
    REMOVE_MARKERS = [
        "INTERACTIVE 3D VIEWER",
        "INTERACTIVE MOLECULAR VIEWER",   # older marker variant
        "Interactive 3D Molecular Viewers",
        "Static protein structure images",
        "Show static PNG images generated by generate_protein_medical_viz",
        "43a_docked_complex",
        "43b_molecular_surface",
        "43c_protein_panels",
        "py3Dmol viewer",
    ]
    def _should_remove(cell):
        return any(m in cell.source for m in REMOVE_MARKERS)
    nb.cells = [c for c in nb.cells if not _should_remove(c)]

    lig_hints = {"ESR1":"4-hydroxytamoxifen (tamoxifen)","ERBB2":"HKI-272 (neratinib analog)","PIK3CA":"inhibitor 1LT"}
    pdb_files  = {"ESR1":"3ERT.pdb","ERBB2":"2JIV.pdb","PIK3CA":"4JPS.pdb"}

    new_cells = []
    new_cells.append(nbformat.v4.new_markdown_cell(
        "## Interactive 3D Molecular Viewers  (py3Dmol / WebGL)\n\n"
        "Run these cells to view **PyMOL-quality** protein-ligand structures "
        "interactively in the notebook.  Rotate / zoom with the mouse.\n\n"
        "**Libraries used for professional molecular visualization:**\n"
        "- `py3Dmol` -- WebGL-based 3D molecular graphics (same renderer as RCSB PDB website)\n"
        "- `pyvista` / VTK -- off-screen PNG export with proper lighting & anti-aliasing\n"
        "- `scikit-image` marching cubes -- Van der Waals molecular surface\n\n"
        "> Visualization style: cartoon ribbon (N-terminus blue --> C-terminus red) + "
        "ligand sticks/spheres + VDW surface coloured by electrostatic potential "
        "(blue=positive, red=negative, white=neutral)"
    ))
    for pdb, prot, lig_rns, sub, ch, *_ in TARGETS:
        cell_code = PY3DMOL_CELL.format(
            prot=prot, subtype=sub, pdb=pdb_files[prot],
            chain=ch, lig_hint=lig_hints[prot])
        new_cells.append(nbformat.v4.new_code_cell(source=cell_code))

    static_code = """\
# Show static PNG images generated by generate_protein_medical_viz.py
from IPython.display import display, HTML
from pathlib import Path

FIG = Path("d:/Aakanksha/thesis/onco-fusion/figures")
for prot in ["ESR1","ERBB2","PIK3CA"]:
    fa = FIG / f"43a_docked_complex_{prot}.png"
    fb = FIG / f"43b_molecular_surface_{prot}.png"
    if fa.exists() and fb.exists():
        display(HTML(
            f"<h4 style='font-family:sans-serif'>{prot}</h4>"
            f"<table><tr>"
            f'<td style="text-align:center"><img src="{fa}" width="420"/>'
            f'<br><i>(a) Docked complex</i></td>'
            f'<td style="text-align:center"><img src="{fb}" width="420"/>'
            f'<br><i>(b) Molecular surface</i></td>'
            f"</tr></table>"
        ))
"""
    new_cells.append(nbformat.v4.new_code_cell(source=static_code))

    nb.cells.extend(new_cells)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"\nNotebook updated: {nb_path.name}  ({len(new_cells)} cells added/refreshed)")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Protein Medical Viz  --  PyVista/VTK + py3Dmol renderer")
    print("  (matplotlib 3D replaced; VTK gives real depth/lighting)")
    print("=" * 65)

    for pdb_file, prot, lig_rns, sub, chain, vv, pr in TARGETS:
        pdb_path = PDB_DIR / pdb_file
        if not pdb_path.exists():
            print(f"\nWARNING: {pdb_file} not found -- skip {prot}")
            continue
        try:
            make_protein_images(pdb_path, prot, lig_rns, chain, sub,
                                view_vec=vv, pocket_radius=pr)
        except Exception as exc:
            import traceback; traceback.print_exc()

    make_combined_panel()
    update_notebook()
    print("\nDone.")


if __name__ == "__main__":
    main()
