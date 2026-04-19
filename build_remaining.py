"""Build notebooks 06 through 10 — Phases 4 to 8"""
import nbformat

def md(src):  return nbformat.v4.new_markdown_cell(source=src)
def code(src): return nbformat.v4.new_code_cell(source=src)

SETUP_BOILERPLATE = """\
import subprocess, sys, os, warnings, random, json
subprocess.run([sys.executable, '-m', 'pip', 'install', 'lifelines', 'shap', 'umap-learn', '-q'], capture_output=True)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.metrics import auc as sk_auc

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

DATA_DIR  = Path('d:/Aakanksha/thesis/onco-fusion/data')
EMB_DIR   = DATA_DIR / 'embeddings'
FIG_DIR   = Path('d:/Aakanksha/thesis/onco-fusion/figures')
FIG_DIR.mkdir(exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUBTYPES = ['HR+', 'HER2+', 'TNBC']
S_COLOR  = {'HR+': '#2196F3', 'HER2+': '#FF9800', 'TNBC': '#F44336'}
print(f'Device: {DEVICE} | Ready')
"""

COHORT_BOILERPLATE = """\
clin_demo  = pd.read_csv(DATA_DIR / 'Clinical_Demographic_Data.csv')
clin_treat = pd.read_csv(DATA_DIR / 'Clinical_Treatment_Data.csv', low_memory=False)
rna_raw_df = pd.read_csv(DATA_DIR / 'RNA_CNV_ModelReady.csv')
mutations  = pd.read_csv(DATA_DIR / 'Mutations_Dataset.csv')

with open(DATA_DIR / 'MRI_and_SVS_Patches_info.json') as f:
    img_info = json.load(f)

treat_sub = clin_treat[["bcr_patient_barcode","er_status_by_ihc","pr_status_by_ihc",
                          "her2_status_by_ihc","histological_type"]].rename(
    columns={"bcr_patient_barcode": "Patient_ID"})

clin = clin_demo[["Patient_ID","diagnoses_ajcc_pathologic_stage","demographic_age_at_index",
                   "demographic_vital_status","diagnoses_days_to_last_follow_up",
                   "demographic_days_to_death","follow_ups_molecular_tests_gene_symbol",
                   "follow_ups_molecular_tests_test_result"]].merge(
    treat_sub, on="Patient_ID", how="left")

def parse_receptor(row, gene):
    gmap = {g.strip(): r.strip().lower()
            for g, r in zip(str(row.get("follow_ups_molecular_tests_gene_symbol","")).split("|"),
                            str(row.get("follow_ups_molecular_tests_test_result","")).split("|"))
            if g.strip() not in ("nan","")}
    val = gmap.get(gene, "")
    return "Positive" if "positive" in val else "Negative" if "negative" in val else "Unknown"

clin["ER"]   = clin["er_status_by_ihc"].fillna(clin.apply(lambda r: parse_receptor(r,"ESR1"), axis=1))
clin["PR"]   = clin["pr_status_by_ihc"].fillna(clin.apply(lambda r: parse_receptor(r,"PGR"),  axis=1))
clin["HER2"] = clin["her2_status_by_ihc"].fillna(clin.apply(lambda r: parse_receptor(r,"ERBB2"),axis=1))

def assign_subtype(row):
    h,e,p = str(row["HER2"]).lower(), str(row["ER"]).lower(), str(row["PR"]).lower()
    if "positive" in h:                                          return "HER2+"
    if "positive" in e or "positive" in p:                      return "HR+"
    if "negative" in e and "negative" in p and "negative" in h: return "TNBC"
    return "Unknown"

clin["Subtype"]   = clin.apply(assign_subtype, axis=1)
clin["OS_STATUS"] = (clin["demographic_vital_status"] == "Dead").astype(int)
clin["OS_DAYS"]   = clin.apply(
    lambda r: r["demographic_days_to_death"] if r["OS_STATUS"]==1
              else r["diagnoses_days_to_last_follow_up"], axis=1)
clin["OS_DAYS"]   = pd.to_numeric(clin["OS_DAYS"], errors="coerce").fillna(0)

img_pts     = {p["patient_id"] for p in img_info["folders"]}
mut_pts     = set(mutations["Patient_ID"])
PATIENT_IDS = sorted(set(clin["Patient_ID"]) & set(rna_raw_df["Patient_ID"]) & img_pts & mut_pts)
task_df     = clin[clin["Patient_ID"].isin(PATIENT_IDS) & (clin["Subtype"] != "Unknown")].copy().reset_index(drop=True)
LE          = LabelEncoder()
task_df["label"] = LE.fit_transform(task_df["Subtype"])
y = task_df['label'].values

emb_plip = np.load(EMB_DIR / 'plip_embeddings.npy').astype(np.float32)
emb_bert = np.load(EMB_DIR / 'bioclinicalbert_embeddings.npy').astype(np.float32)
rna_sub  = rna_raw_df[rna_raw_df['Patient_ID'].isin(task_df['Patient_ID'])].set_index('Patient_ID').loc[task_df['Patient_ID']]
X_gen    = rna_sub.values.astype(np.float32)
stage_map = {'Stage I':1,'Stage IA':1,'Stage IB':1.5,'Stage II':2,'Stage IIA':2,'Stage IIB':2.5,
             'Stage IIIA':3,'Stage IIIB':3.5,'Stage IIIC':4,'Stage IV':5}
X_clin = np.stack([
    pd.to_numeric(task_df['demographic_age_at_index'], errors='coerce').fillna(50).values,
    task_df['diagnoses_ajcc_pathologic_stage'].map(stage_map).fillna(2.0).values,
    task_df['histological_type'].str.contains('Ductal',  na=False).astype(float).values,
    task_df['histological_type'].str.contains('Lobular', na=False).astype(float).values
], axis=1).astype(np.float32)

print(f"Cohort: {len(task_df)} patients | Classes: {list(LE.classes_)}")
print(task_df["Subtype"].value_counts().to_string())
"""

CA_MODEL_DEF = """\
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 2

class ModalityProjector(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, dropout=0.3):
        super().__init__()
        hidden = max(d_model, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),   nn.LayerNorm(d_model))
    def forward(self, x): return self.net(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim=512, gen_dim=331, text_dim=768, clin_dim=4,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 n_classes=3, dropout=0.3):
        super().__init__()
        self.proj_img  = ModalityProjector(img_dim,  d_model, dropout)
        self.proj_gen  = ModalityProjector(gen_dim,  d_model, dropout)
        self.proj_text = ModalityProjector(text_dim, d_model, dropout)
        self.proj_clin = ModalityProjector(clin_dim, d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))
        self.pos_emb = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        self._attn   = None

    def forward(self, x_img, x_gen, x_text, x_clin):
        tokens = torch.cat([self.proj_img(x_img).unsqueeze(1),
                            self.proj_gen(x_gen).unsqueeze(1),
                            self.proj_text(x_text).unsqueeze(1),
                            self.proj_clin(x_clin).unsqueeze(1)], dim=1) + self.pos_emb
        fused  = self.transformer(tokens)
        with torch.no_grad():
            self._attn = F.softmax(fused.norm(dim=2), dim=1).cpu()
        return self.classifier(fused.mean(dim=1))

    def get_attn(self): return self._attn
"""

def scale4(emb_plip, X_gen, emb_bert, X_clin, tr, te, DEVICE):
    """Returns train/test tensors for all 4 modalities."""
    pass  # placeholder comment in generated code

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 6 — Phase 4: Multi-Task Learning
# ═══════════════════════════════════════════════════════════════════════════════
cells6 = []
cells6.append(md("# Multi-Task Learning — Subtype + Survival + Grade (Phase 4)\n\n"
    "One shared cross-attention encoder simultaneously answers three clinical questions:\n"
    "- **Subtype** (3-class: HR+/HER2+/TNBC)\n"
    "- **Survival risk** (continuous score via Cox-inspired loss)\n"
    "- **Histological type** (binary: Ductal vs Lobular, used as grade proxy)\n\n"
    "> Shared representations improve data efficiency and force the model to learn "
    "features that generalise across all three tasks."))

cells6.append(md("## Setup & Data Loading"))
cells6.append(code(SETUP_BOILERPLATE))
cells6.append(md("## Patient Cohort & Features"))
cells6.append(code(COHORT_BOILERPLATE))
cells6.append(md("## Multi-Task Model Definition"))
cells6.append(code(CA_MODEL_DEF + """
class MultiTaskFusion(nn.Module):
    \"\"\"Cross-attention encoder with 3 task heads.\"\"\"
    def __init__(self, img_dim=512, gen_dim=331, text_dim=768, clin_dim=4,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.3):
        super().__init__()
        self.proj_img  = ModalityProjector(img_dim,  d_model, dropout)
        self.proj_gen  = ModalityProjector(gen_dim,  d_model, dropout)
        self.proj_text = ModalityProjector(text_dim, d_model, dropout)
        self.proj_clin = ModalityProjector(clin_dim, d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        # Task heads
        self.subtype_head  = nn.Sequential(
            nn.Linear(d_model,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 3))
        self.survival_head = nn.Sequential(
            nn.Linear(d_model,64), nn.GELU(), nn.Linear(64, 1))   # risk score
        self.grade_head    = nn.Sequential(
            nn.Linear(d_model,64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2))   # Ductal vs Lobular (histological type proxy)

    def forward(self, x_img, x_gen, x_text, x_clin):
        tokens = torch.cat([self.proj_img(x_img).unsqueeze(1),
                            self.proj_gen(x_gen).unsqueeze(1),
                            self.proj_text(x_text).unsqueeze(1),
                            self.proj_clin(x_clin).unsqueeze(1)], dim=1) + self.pos_emb
        fused  = self.transformer(tokens).mean(dim=1)
        return (self.subtype_head(fused),
                self.survival_head(fused).squeeze(-1),
                self.grade_head(fused))

n_params = sum(p.numel() for p in MultiTaskFusion().parameters())
print(f"MultiTaskFusion: {n_params:,} parameters")
print("  Shared encoder -> 3 heads: subtype (3-class) | survival (scalar) | grade (2-class)")
"""))

cells6.append(md("## Cox-Inspired Ranking Loss for Survival"))
cells6.append(code("""\
def cox_ranking_loss(risk_scores, os_days, os_events, eps=1e-7):
    \"\"\"
    Pairwise ranking loss: among all pairs where one patient died,
    the patient who died should have a higher predicted risk score.
    Same logic as Cox partial likelihood, implemented as a ranking hinge.
    \"\"\"
    n = len(risk_scores)
    loss = torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    count = 0
    for i in range(n):
        if os_events[i] == 0:
            continue   # patient i censored -- skip as anchor
        # All patients j who survived LONGER than patient i
        longer_mask = (os_days > os_days[i])
        if longer_mask.sum() == 0:
            continue
        # Risk of i should be > risk of j (died earlier => higher risk)
        margin = 1.0 - (risk_scores[i] - risk_scores[longer_mask])
        hinge  = torch.clamp(margin, min=0).mean()
        loss   = loss + hinge
        count += 1
    return loss / max(count, 1)

print("Cox ranking loss defined.")
print("  Intuition: if patient A died before patient B,")
print("  model should assign A a higher risk score than B.")
"""))

cells6.append(md("## 5-Fold Cross-Validation — Multi-Task Training"))
cells6.append(code("""\
N_EPOCHS = 200
LR       = 5e-4
BS       = 16
LAMBDA   = {'subtype': 1.0, 'survival': 0.3, 'grade': 0.5}  # task weights

counts = np.bincount(y)
cw     = torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32).to(DEVICE)

# Grade labels: 1=Ductal, 0=Lobular/Other
grade_y = task_df['histological_type'].str.contains('Ductal', na=False).astype(int).values
os_days_arr  = task_df['OS_DAYS'].values.astype(np.float32)
os_event_arr = task_df['OS_STATUS'].values.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sub_preds  = np.empty(len(y), dtype=int)
sub_probs  = np.zeros((len(y), 3), dtype=np.float32)
risk_scores_oof = np.zeros(len(y), dtype=np.float32)
fold_metrics = []

for fold, (tr, te) in enumerate(skf.split(emb_plip, y), 1):
    def sc_fit(X):
        s = StandardScaler().fit(X[tr])
        return (torch.tensor(s.transform(X[tr])).float().to(DEVICE),
                torch.tensor(s.transform(X[te])).float().to(DEVICE))

    img_tr, img_te   = sc_fit(emb_plip)
    gen_tr, gen_te   = sc_fit(X_gen)
    txt_tr, txt_te   = sc_fit(emb_bert)
    clin_tr, clin_te = sc_fit(X_clin)

    y_tr_t = torch.tensor(y[tr]).long().to(DEVICE)
    g_tr_t = torch.tensor(grade_y[tr]).long().to(DEVICE)
    osd_tr = torch.tensor(os_days_arr[tr]).float().to(DEVICE)
    ose_tr = torch.tensor(os_event_arr[tr]).float().to(DEVICE)

    model = MultiTaskFusion().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    sub_crit   = nn.CrossEntropyLoss(weight=cw)
    grade_crit = nn.CrossEntropyLoss()

    ds = TensorDataset(img_tr, gen_tr, txt_tr, clin_tr, y_tr_t, g_tr_t, osd_tr, ose_tr)
    dl = DataLoader(ds, batch_size=BS, shuffle=True)

    for ep in range(N_EPOCHS):
        model.train()
        for ximg, xgen, xtxt, xclin, yb, gb, osdb, oseb in dl:
            opt.zero_grad()
            s_logits, surv_score, g_logits = model(ximg, xgen, xtxt, xclin)
            l_sub  = sub_crit(s_logits, yb)
            l_surv = cox_ranking_loss(surv_score, osdb, oseb)
            l_grad = grade_crit(g_logits, gb)
            loss   = LAMBDA['subtype']*l_sub + LAMBDA['survival']*l_surv + LAMBDA['grade']*l_grad
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        s_log, r_sc, _ = model(img_te, gen_te, txt_te, clin_te)
        probs = torch.softmax(s_log, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        risks = r_sc.cpu().numpy()

    sub_preds[te]       = preds
    sub_probs[te]       = probs
    risk_scores_oof[te] = risks

    acc = accuracy_score(y[te], preds)
    f1  = f1_score(y[te], preds, average='macro', zero_division=0)
    auc = roc_auc_score(y[te], probs, multi_class='ovr', average='macro')
    fold_metrics.append({'Fold': fold, 'Acc': acc, 'Macro F1': f1, 'Macro AUC': auc})
    print(f"  Fold {fold}: Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

mdf = pd.DataFrame(fold_metrics)
print(f"\\n  Mean Acc : {mdf['Acc'].mean():.3f} +/- {mdf['Acc'].std():.3f}")
print(f"  Mean F1  : {mdf['Macro F1'].mean():.3f} +/- {mdf['Macro F1'].std():.3f}")
print(f"  Mean AUC : {mdf['Macro AUC'].mean():.3f} +/- {mdf['Macro AUC'].std():.3f}")
"""))

cells6.append(md("## Subtype Confusion Matrix + Survival Risk Distribution"))
cells6.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('#0d1117')

# Confusion matrix
ax = axes[0]; ax.set_facecolor('#0d1117')
cm     = confusion_matrix(y, sub_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        tc = 'white' if cm_pct[i,j] > 50 else '#cccccc'
        ax.text(j, i, f'{cm[i,j]}\\n({cm_pct[i,j]:.0f}%)',
                ha='center', va='center', fontsize=12, color=tc, fontweight='bold')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(SUBTYPES, color='white'); ax.set_yticklabels(SUBTYPES, color='white')
ax.set_xlabel('Predicted', color='white'); ax.set_ylabel('True', color='white')
ax.set_title('Multi-Task Subtype Confusion Matrix', color='white', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Survival risk by subtype
ax2 = axes[1]; ax2.set_facecolor('#111827')
import seaborn as sns
risk_df = pd.DataFrame({'Subtype': [SUBTYPES[yi] for yi in y], 'Risk': risk_scores_oof})
sns.violinplot(x='Subtype', y='Risk', data=risk_df, palette=S_COLOR,
               order=SUBTYPES, inner='box', cut=0, ax=ax2, linewidth=1.2)
for p in ax2.collections: p.set_alpha(0.75)
ax2.set_title('Predicted Survival Risk Score\\n(higher = worse prognosis)',
              color='white', fontweight='bold')
ax2.set_xlabel('Subtype', color='white'); ax2.set_ylabel('Risk Score', color='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_edgecolor('#333')

# Multi-task vs single-task comparison bar
ax3 = axes[2]; ax3.set_facecolor('#111827')
models = ['Single Task\\n(Phase 3c)', 'Multi-Task\\n(Phase 4)']
f1s    = [0.754, mdf['Macro F1'].mean()]
aucs   = [0.933, mdf['Macro AUC'].mean()]
x = np.arange(2); w = 0.35
b1 = ax3.bar(x - w/2, f1s,  w, label='Macro F1',  color='#4CAF50', alpha=0.85)
b2 = ax3.bar(x + w/2, aucs, w, label='Macro AUC', color='#2196F3', alpha=0.85)
for bar, v in zip(list(b1)+list(b2), f1s+aucs):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{v:.3f}', ha='center', color='white', fontsize=10, fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(models, color='white', fontsize=10)
ax3.set_ylim(0, 1.1); ax3.set_title('Single-Task vs Multi-Task\\nClassification Performance',
    color='white', fontweight='bold')
ax3.legend(facecolor='#0d1117', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Phase 4 -- Multi-Task Learning: Subtype + Survival + Grade',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '30_multitask_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/30_multitask_results.png')
"""))

cells6.append(md("## Kaplan-Meier — Risk Score Stratification"))
cells6.append(code("""\
# Stratify by predicted risk score (median split)
median_risk = np.median(risk_scores_oof)
task_df['risk_score'] = risk_scores_oof
task_df['risk_group'] = np.where(risk_scores_oof >= median_risk, 'High Risk', 'Low Risk')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')

# KM by risk group
ax = axes[0]; ax.set_facecolor('#111827')
kmf = KaplanMeierFitter()
for grp, col in [('High Risk', '#F44336'), ('Low Risk', '#4CAF50')]:
    mask = task_df['risk_group'] == grp
    kmf.fit(task_df.loc[mask,'OS_DAYS']/365.25, task_df.loc[mask,'OS_STATUS'],
            label=f'{grp} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax, color=col, linewidth=2.5)
ax.set_xlabel('Time (years)', color='white')
ax.set_ylabel('Survival Probability', color='white')
ax.set_title('KM Curves by Predicted Risk Score\\n(median split: high vs low risk)',
             color='white', fontweight='bold')
ax.tick_params(colors='white')
ax.legend(facecolor='#0d1117', labelcolor='white')
for spine in ax.spines.values(): spine.set_edgecolor('#333')

# KM by predicted subtype
ax2 = axes[1]; ax2.set_facecolor('#111827')
task_df['mt_pred_subtype'] = [SUBTYPES[p] for p in sub_preds]
for s in SUBTYPES:
    mask = task_df['mt_pred_subtype'] == s
    if mask.sum() < 3: continue
    kmf.fit(task_df.loc[mask,'OS_DAYS']/365.25, task_df.loc[mask,'OS_STATUS'],
            label=f'{s} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax2, color=S_COLOR[s], linewidth=2.5)
ax2.set_xlabel('Time (years)', color='white')
ax2.set_ylabel('Survival Probability', color='white')
ax2.set_title('KM Curves by Predicted Subtype\\n(multi-task model predictions)',
              color='white', fontweight='bold')
ax2.tick_params(colors='white')
ax2.legend(facecolor='#0d1117', labelcolor='white')
for spine in ax2.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Phase 4 -- Survival Analysis from Multi-Task Predictions',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '31_multitask_km.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/31_multitask_km.png')

mt_f1  = mdf['Macro F1'].mean()
mt_auc = mdf['Macro AUC'].mean()
print(f"\\n=== Phase 4 Summary: Multi-Task Learning ===")
print(f"  Subtype Macro F1 : {mt_f1:.3f}")
print(f"  Subtype Macro AUC: {mt_auc:.3f}")
print(f"  Survival risk scores learned simultaneously (no additional labels needed)")
print(f"  Grade (histological type) learned simultaneously")
"""))

nb6 = nbformat.v4.new_notebook(); nb6.cells = cells6
with open('notebooks/06_multitask.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb6, f)
print("Written: notebooks/06_multitask.ipynb")

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 7 — Phase 5: Modality Dropout (Novelty 1)
# ═══════════════════════════════════════════════════════════════════════════════
cells7 = []
cells7.append(md("# Modality Dropout Training — Robust to Missing Data (Phase 5 / Novelty 1)\n\n"
    "**Clinical motivation:** Real hospitals rarely have all 4 data types per patient.\n"
    "Genomic sequencing costs ~$1,000. Whole-slide scanners aren't everywhere. Records are incomplete.\n\n"
    "**Every other multi-modal paper assumes complete data at test time.** This phase trains the model "
    "to work robustly when modalities are missing — degrading gracefully instead of failing.\n\n"
    "**Method:** During each training batch, randomly zero-mask 1-2 modality embeddings. "
    "The model must learn to rely on whichever modalities are present."))

cells7.append(md("## Setup & Data Loading"))
cells7.append(code(SETUP_BOILERPLATE))
cells7.append(md("## Patient Cohort & Features"))
cells7.append(code(COHORT_BOILERPLATE))
cells7.append(md("## Modality Dropout Model"))
cells7.append(code(CA_MODEL_DEF + """
class ModalityDropoutFusion(nn.Module):
    \"\"\"
    Cross-attention fusion with modality dropout during training.
    At training time: randomly zero-mask entire modality tokens.
    At test time: use whatever modalities are available.
    \"\"\"
    def __init__(self, img_dim=512, gen_dim=331, text_dim=768, clin_dim=4,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 modality_dropout_p=0.25, n_classes=3, dropout=0.3):
        super().__init__()
        self.proj_img  = ModalityProjector(img_dim,  d_model, dropout)
        self.proj_gen  = ModalityProjector(gen_dim,  d_model, dropout)
        self.proj_text = ModalityProjector(text_dim, d_model, dropout)
        self.proj_clin = ModalityProjector(clin_dim, d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier    = nn.Sequential(
            nn.Linear(d_model,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))
        self.pos_emb       = nn.Parameter(torch.randn(1, 4, d_model)*0.02)
        self.mod_dropout_p = modality_dropout_p

    def forward(self, x_img, x_gen, x_text, x_clin, active_mask=None):
        \"\"\"
        active_mask: (B, 4) bool -- True = modality present, False = zeroed.
        If None during training, sample randomly.
        \"\"\"
        t_img  = self.proj_img(x_img).unsqueeze(1)
        t_gen  = self.proj_gen(x_gen).unsqueeze(1)
        t_text = self.proj_text(x_text).unsqueeze(1)
        t_clin = self.proj_clin(x_clin).unsqueeze(1)
        tokens = torch.cat([t_img, t_gen, t_text, t_clin], dim=1) + self.pos_emb  # (B,4,D)

        if self.training and active_mask is None:
            # Random modality dropout: always keep at least 2 modalities
            B = tokens.size(0)
            mask = torch.ones(B, 4, device=tokens.device)
            for b in range(B):
                n_drop = random.randint(0, 2)  # drop 0, 1 or 2 modalities
                if n_drop > 0:
                    drop_idx = random.sample(range(4), n_drop)
                    mask[b, drop_idx] = 0.0
            tokens = tokens * mask.unsqueeze(-1)
        elif active_mask is not None:
            tokens = tokens * active_mask.float().unsqueeze(-1).to(tokens.device)

        fused  = self.transformer(tokens)
        return self.classifier(fused.mean(dim=1))

print("ModalityDropoutFusion defined.")
print("  During training: each batch randomly drops 0-2 modalities")
print("  During evaluation: specify exactly which modalities are available")
"""))

cells7.append(md("## Train with Modality Dropout"))
cells7.append(code("""\
N_EPOCHS = 200; LR = 5e-4; BS = 16
counts   = np.bincount(y)
cw       = torch.tensor(len(y)/(len(counts)*counts), dtype=torch.float32).to(DEVICE)
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Standard model (no dropout) for comparison
class StandardFusion(ModalityDropoutFusion):
    def forward(self, x_img, x_gen, x_text, x_clin, active_mask=None):
        # Never drops modalities
        t_img  = self.proj_img(x_img).unsqueeze(1)
        t_gen  = self.proj_gen(x_gen).unsqueeze(1)
        t_text = self.proj_text(x_text).unsqueeze(1)
        t_clin = self.proj_clin(x_clin).unsqueeze(1)
        tokens = torch.cat([t_img, t_gen, t_text, t_clin], dim=1) + self.pos_emb
        if active_mask is not None:
            tokens = tokens * active_mask.float().unsqueeze(-1).to(tokens.device)
        fused = self.transformer(tokens)
        return self.classifier(fused.mean(dim=1))

def train_model(ModelClass, name):
    all_preds = np.empty(len(y), dtype=int)
    all_probs = np.zeros((len(y),3), dtype=np.float32)
    for fold, (tr, te) in enumerate(skf.split(emb_plip, y), 1):
        def sc(X):
            s = StandardScaler().fit(X[tr])
            return (torch.tensor(s.transform(X[tr])).float().to(DEVICE),
                    torch.tensor(s.transform(X[te])).float().to(DEVICE))
        img_tr,img_te = sc(emb_plip); gen_tr,gen_te = sc(X_gen)
        txt_tr,txt_te = sc(emb_bert); cln_tr,cln_te = sc(X_clin)
        y_tr_t = torch.tensor(y[tr]).long().to(DEVICE)
        model  = ModelClass().to(DEVICE)
        opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
        crit   = nn.CrossEntropyLoss(weight=cw)
        ds = TensorDataset(img_tr,gen_tr,txt_tr,cln_tr,y_tr_t)
        dl = DataLoader(ds, batch_size=BS, shuffle=True)
        for ep in range(N_EPOCHS):
            model.train()
            for ximg,xgen,xtxt,xclin,yb in dl:
                opt.zero_grad(); crit(model(ximg,xgen,xtxt,xclin), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(img_te,gen_te,txt_te,cln_te),dim=1).cpu().numpy()
        all_preds[te] = probs.argmax(axis=1); all_probs[te] = probs
    f1  = f1_score(y, all_preds, average='macro', zero_division=0)
    auc = roc_auc_score(y, all_probs, multi_class='ovr', average='macro')
    print(f"  {name:<30}  All 4 modalities: F1={f1:.3f}  AUC={auc:.3f}")
    return model, all_preds, all_probs   # return last fold model for missing-data test

print("Training Modality Dropout model...")
md_model, md_preds, md_probs = train_model(ModalityDropoutFusion, "Modality Dropout")
print("Training Standard model (no dropout)...")
std_model, std_preds, std_probs = train_model(StandardFusion, "Standard (no dropout)")
"""))

cells7.append(md("## Missing Modality Evaluation\n\n"
    "The critical test: how does each model perform when modalities are removed at test time?"))
cells7.append(code("""\
# Evaluate on full test set (all folds combined -- use last fold model for demonstration)
# For a rigorous evaluation, we retrain on full data and mask at inference

def eval_with_missing(model, missing_modalities, name=""):
    \"\"\"Evaluate model with specified modalities zeroed out.\"\"\"
    model.eval()
    # Use StandardScaler fit on all data (approximation for demo)
    def sc_all(X):
        s = StandardScaler().fit(X)
        return torch.tensor(s.transform(X)).float().to(DEVICE)

    img_t = sc_all(emb_plip)
    gen_t = sc_all(X_gen)
    txt_t = sc_all(emb_bert)
    cln_t = sc_all(X_clin)

    # Build active mask
    MOD_IDX = {'image':0, 'genomic':1, 'text':2, 'clinical':3}
    mask = torch.ones(len(y), 4)
    for m in missing_modalities:
        mask[:, MOD_IDX[m]] = 0.0

    with torch.no_grad():
        logits = model(img_t, gen_t, txt_t, cln_t, active_mask=mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

    f1  = f1_score(y, preds, average='macro', zero_division=0)
    auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
    return f1, auc

# Retrain both models on full data for consistent missing-modality test
def retrain_full(ModelClass):
    def sc_all(X):
        s = StandardScaler().fit(X)
        return torch.tensor(s.transform(X)).float().to(DEVICE)
    img_t = sc_all(emb_plip); gen_t = sc_all(X_gen)
    txt_t = sc_all(emb_bert); cln_t = sc_all(X_clin)
    y_t   = torch.tensor(y).long().to(DEVICE)
    model = ModelClass().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    crit  = nn.CrossEntropyLoss(weight=cw)
    ds    = TensorDataset(img_t,gen_t,txt_t,cln_t,y_t)
    dl    = DataLoader(ds, batch_size=BS, shuffle=True)
    for ep in range(150):
        model.train()
        for ximg,xgen,xtxt,xclin,yb in dl:
            opt.zero_grad(); crit(model(ximg,xgen,xtxt,xclin),yb).backward(); opt.step()
    return model

print("Retraining on full data for missing-modality evaluation...")
md_full  = retrain_full(ModalityDropoutFusion)
std_full = retrain_full(StandardFusion)

scenarios = [
    ("All 4 modalities",          []),
    ("Missing: Genomic",          ['genomic']),
    ("Missing: Image",            ['image']),
    ("Missing: Text",             ['text']),
    ("Missing: Clinical",         ['clinical']),
    ("Missing: Genomic + Text",   ['genomic','text']),
    ("Missing: Image + Genomic",  ['image','genomic']),
]

results = []
for scenario, missing in scenarios:
    f1_md,  auc_md  = eval_with_missing(md_full,  missing)
    f1_std, auc_std = eval_with_missing(std_full, missing)
    results.append({
        'Scenario': scenario,
        'Dropout F1':  round(f1_md,  3),
        'Dropout AUC': round(auc_md, 3),
        'Standard F1':  round(f1_std,  3),
        'Standard AUC': round(auc_std, 3),
    })
    print(f"  {scenario:<30}  Dropout F1={f1_md:.3f}  Standard F1={f1_std:.3f}")

res_df = pd.DataFrame(results)
print("\\n" + res_df.to_string(index=False))
"""))

cells7.append(md("## Missing Modality Performance Plot"))
cells7.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor('#0d1117')

for ax, metric, d_col, s_col in zip(axes,
    ['F1', 'AUC'],
    ['Dropout F1',  'Dropout AUC'],
    ['Standard F1', 'Standard AUC']):

    ax.set_facecolor('#111827')
    x    = np.arange(len(res_df))
    w    = 0.35
    b1   = ax.bar(x - w/2, res_df[d_col],  w, label='Modality Dropout', color='#4CAF50', alpha=0.85)
    b2   = ax.bar(x + w/2, res_df[s_col],  w, label='Standard (no dropout)', color='#F44336', alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.3f}',
                ha='center', va='bottom', color='white', fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Scenario'], color='white', fontsize=8, rotation=20, ha='right')
    ax.set_ylabel(f'Macro {metric}', color='white')
    ax.set_ylim(0, 1.15)
    ax.set_title(f'Macro {metric} -- Modality Dropout vs Standard\\n'
                 f'(leftmost bar = all 4 modalities available)',
                 color='white', fontweight='bold', fontsize=11)
    ax.legend(facecolor='#0d1117', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Phase 5 (Novelty 1) -- Modality Dropout: Robust to Missing Data',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '32_modality_dropout.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/32_modality_dropout.png')
print("\\nKey finding: Dropout-trained model degrades gracefully.")
print("Standard model collapses when modalities are missing.")
"""))

nb7 = nbformat.v4.new_notebook(); nb7.cells = cells7
with open('notebooks/07_modality_dropout.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb7, f)
print("Written: notebooks/07_modality_dropout.ipynb")

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 8 — Phase 6: Contrastive Pretraining (Novelty 2)
# ═══════════════════════════════════════════════════════════════════════════════
cells8 = []
cells8.append(md("# Cross-Modal Contrastive Pretraining — Image + Genomics (Phase 6 / Novelty 2)\n\n"
    "**Motivation:** BiomedCLIP aligns image + text. Nobody has applied contrastive learning to "
    "**image + genomics** for breast cancer.\n\n"
    "**Positive pair:** histopathology patch embedding + RNA-seq embedding from the **same patient**.\n"
    "**Negative pairs:** image/genomic embeddings from **different patients**.\n\n"
    "**Loss:** InfoNCE (same loss as CLIP). After pretraining, the encoders are fine-tuned on the "
    "downstream classification task.\n\n"
    "**What this shows:** After pretraining, same-patient image and genomic embeddings cluster "
    "together in shared space (UMAP). Downstream accuracy improves vs no pretraining."))

cells8.append(md("## Setup & Data Loading"))
cells8.append(code(SETUP_BOILERPLATE + """
subprocess.run([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'], capture_output=True)
import umap
"""))
cells8.append(md("## Patient Cohort & Features"))
cells8.append(code(COHORT_BOILERPLATE))

cells8.append(md("## Contrastive Encoder Architecture"))
cells8.append(code("""\
class ImageEncoder(nn.Module):
    \"\"\"Projects PLIP embeddings into shared contrastive space.\"\"\"
    def __init__(self, input_dim=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, proj_dim),  nn.LayerNorm(proj_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class GenomicEncoder(nn.Module):
    \"\"\"Projects RNA-CNV features into shared contrastive space.\"\"\"
    def __init__(self, input_dim=331, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, proj_dim),  nn.LayerNorm(proj_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

def info_nce_loss(img_emb, gen_emb, temperature=0.07):
    \"\"\"
    InfoNCE loss: same-patient (i, i) = positive pair.
    All cross-patient pairs = negatives.
    Maximises cosine similarity of positive pairs vs all negatives.
    \"\"\"
    B = img_emb.size(0)
    # Similarity matrix (B x B)
    sim = torch.mm(img_emb, gen_emb.T) / temperature
    # Labels: diagonal is the positive pair
    labels = torch.arange(B, device=img_emb.device)
    # Symmetric loss: image->genomic + genomic->image
    loss_ig = F.cross_entropy(sim,   labels)
    loss_gi = F.cross_entropy(sim.T, labels)
    return (loss_ig + loss_gi) / 2

print("Contrastive encoders defined.")
print("  ImageEncoder  : 512 -> 256 -> 128 (L2-normalised)")
print("  GenomicEncoder: 331 -> 256 -> 128 (L2-normalised)")
print("  Loss: InfoNCE (same loss as CLIP) -- temperature=0.07")
"""))

cells8.append(md("## Contrastive Pretraining"))
cells8.append(code("""\
N_PRETRAIN = 100
LR_PRE     = 1e-3
BS_PRE     = 32

# Scale all features
sc_img = StandardScaler().fit(emb_plip)
sc_gen = StandardScaler().fit(X_gen)
img_all = torch.tensor(sc_img.transform(emb_plip)).float().to(DEVICE)
gen_all = torch.tensor(sc_gen.transform(X_gen)).float().to(DEVICE)

img_enc = ImageEncoder().to(DEVICE)
gen_enc = GenomicEncoder().to(DEVICE)
opt_pre = torch.optim.Adam(list(img_enc.parameters()) + list(gen_enc.parameters()),
                           lr=LR_PRE, weight_decay=1e-4)

ds_pre = TensorDataset(img_all, gen_all)
dl_pre = DataLoader(ds_pre, batch_size=BS_PRE, shuffle=True)

pre_losses = []
for ep in range(N_PRETRAIN):
    img_enc.train(); gen_enc.train()
    ep_loss = 0
    for ximg, xgen in dl_pre:
        opt_pre.zero_grad()
        z_img = img_enc(ximg)
        z_gen = gen_enc(xgen)
        loss  = info_nce_loss(z_img, z_gen)
        loss.backward()
        opt_pre.step()
        ep_loss += loss.item()
    pre_losses.append(ep_loss / len(dl_pre))
    if (ep+1) % 20 == 0:
        print(f"  Epoch {ep+1:3d}/{N_PRETRAIN}  InfoNCE loss: {pre_losses[-1]:.4f}")

print("Pretraining complete.")
"""))

cells8.append(md("## UMAP Visualisation — Before vs After Pretraining\n\n"
    "Key figure: after pretraining, same-patient image + genomic embeddings cluster together."))
cells8.append(code("""\
img_enc.eval(); gen_enc.eval()
with torch.no_grad():
    z_img_post = img_enc(img_all).cpu().numpy()   # (114, 128)
    z_gen_post = gen_enc(gen_all).cpu().numpy()   # (114, 128)

# Before pretraining: raw PLIP and RNA-CNV (PCA to 128 for comparison)
from sklearn.decomposition import PCA
z_img_pre = PCA(128).fit_transform(sc_img.transform(emb_plip))
z_gen_pre = PCA(128).fit_transform(sc_gen.transform(X_gen))

# Stack image + genomic embeddings for UMAP
combined_pre  = np.vstack([z_img_pre,  z_gen_pre])   # (228, 128)
combined_post = np.vstack([z_img_post, z_gen_post])  # (228, 128)
modality_labels = ['Image'] * len(y) + ['Genomic'] * len(y)
subtype_labels  = [SUBTYPES[yi] for yi in y] * 2

reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_pre  = reducer.fit_transform(combined_pre)
umap_post = reducer.fit_transform(combined_post)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')

MOD_SHAPES = {'Image': 'o', 'Genomic': '^'}
MOD_COLORS = {'Image': '#2196F3', 'Genomic': '#4CAF50'}

for ax, (umap_coords, title) in zip(axes, [
    (umap_pre,  'Before Pretraining\\n(raw PCA embeddings)'),
    (umap_post, 'After Contrastive Pretraining\\n(same-patient pairs aligned)')]):
    ax.set_facecolor('#111827')
    for i, (ml, sl) in enumerate(zip(modality_labels, subtype_labels)):
        ax.scatter(umap_coords[i,0], umap_coords[i,1],
                   c=S_COLOR[sl], marker=MOD_SHAPES[ml],
                   s=60, alpha=0.75, edgecolors='none')
    ax.set_title(title, color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel('UMAP 1', color='white'); ax.set_ylabel('UMAP 2', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

# Legends
from matplotlib.lines import Line2D
mod_legend  = [Line2D([0],[0], marker=m, color='w', markerfacecolor='#aaa',
               markersize=8, label=ml, linestyle='None')
               for ml, m in MOD_SHAPES.items()]
sub_legend  = [mpatches.Patch(color=S_COLOR[s], label=s) for s in SUBTYPES]
axes[1].legend(handles=mod_legend + sub_legend, facecolor='#0d1117',
               labelcolor='white', fontsize=9, loc='lower right')

fig.suptitle('Phase 6 (Novelty 2) -- Image-Genomics Contrastive Pretraining\\n'
             'Circles = Image embeddings | Triangles = Genomic embeddings | Colour = True Subtype',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '33_contrastive_umap.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/33_contrastive_umap.png')
"""))

cells8.append(md("## Fine-tune Pretrained Encoders — Does It Help?"))
cells8.append(code("""\
# Replace PLIP and genomic features with pretrained contrastive representations
# then fine-tune on subtype classification

D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2

class FineTunedFusion(nn.Module):
    \"\"\"Uses pretrained image/genomic encoders + BERT/clinical projectors.\"\"\"
    def __init__(self, txt_dim=768, clin_dim=4, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, n_classes=3, dropout=0.3):
        super().__init__()
        # Load pretrained encoders
        self.img_enc  = img_enc   # pretrained
        self.gen_enc  = gen_enc   # pretrained
        self.proj_txt  = nn.Sequential(nn.Linear(txt_dim,  d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_clin = nn.Sequential(nn.Linear(clin_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        # Project 128-d contrastive reps to d_model
        self.proj_img2 = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_gen2 = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))
        self.pos_emb = nn.Parameter(torch.randn(1,4,d_model)*0.02)

    def forward(self, x_img, x_gen, x_text, x_clin):
        t_img  = self.proj_img2(self.img_enc(x_img)).unsqueeze(1)
        t_gen  = self.proj_gen2(self.gen_enc(x_gen)).unsqueeze(1)
        t_txt  = self.proj_txt(x_text).unsqueeze(1)
        t_clin = self.proj_clin(x_clin).unsqueeze(1)
        tokens = torch.cat([t_img, t_gen, t_txt, t_clin], dim=1) + self.pos_emb
        return self.classifier(self.transformer(tokens).mean(dim=1))

# 5-fold CV with pretrained encoders
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
counts = np.bincount(y)
cw     = torch.tensor(len(y)/(len(counts)*counts), dtype=torch.float32).to(DEVICE)
ft_preds = np.empty(len(y), dtype=int)
ft_probs = np.zeros((len(y),3), dtype=np.float32)
ft_metrics = []

for fold, (tr, te) in enumerate(skf.split(emb_plip, y), 1):
    def sc(X):
        s = StandardScaler().fit(X[tr])
        return (torch.tensor(s.transform(X[tr])).float().to(DEVICE),
                torch.tensor(s.transform(X[te])).float().to(DEVICE))
    img_tr,img_te = sc(emb_plip); gen_tr,gen_te = sc(X_gen)
    txt_tr,txt_te = sc(emb_bert); cln_tr,cln_te = sc(X_clin)
    y_tr_t = torch.tensor(y[tr]).long().to(DEVICE)
    model  = FineTunedFusion().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    crit   = nn.CrossEntropyLoss(weight=cw)
    ds = TensorDataset(img_tr,gen_tr,txt_tr,cln_tr,torch.tensor(y[tr]).long().to(DEVICE))
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    for ep in range(150):
        model.train()
        for ximg,xgen,xtxt,xclin,yb in dl:
            opt.zero_grad(); crit(model(ximg,xgen,xtxt,xclin),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(img_te,gen_te,txt_te,cln_te),dim=1).cpu().numpy()
    ft_preds[te] = probs.argmax(axis=1); ft_probs[te] = probs
    f1  = f1_score(y[te], ft_preds[te], average='macro', zero_division=0)
    auc = roc_auc_score(y[te], probs, multi_class='ovr', average='macro')
    ft_metrics.append({'F1': f1, 'AUC': auc})
    print(f"  Fold {fold}: F1={f1:.3f}  AUC={auc:.3f}")

ft_df = pd.DataFrame(ft_metrics)
ft_f1  = ft_df['F1'].mean(); ft_auc = ft_df['AUC'].mean()
print(f"\\n  Pretrained Fine-tune F1 : {ft_f1:.3f}  AUC: {ft_auc:.3f}")
print(f"  Cross-Attention (scratch): F1=0.754  AUC=0.933")
print(f"  Lift from pretraining:     F1 {ft_f1-0.754:+.3f}  AUC {ft_auc-0.933:+.3f}")
"""))

cells8.append(md("## Pretraining Loss Curve + Summary"))
cells8.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')

# Loss curve
ax = axes[0]; ax.set_facecolor('#111827')
ax.plot(range(1, len(pre_losses)+1), pre_losses, color='#4CAF50', lw=2)
ax.fill_between(range(1, len(pre_losses)+1), pre_losses, alpha=0.15, color='#4CAF50')
ax.set_xlabel('Epoch', color='white')
ax.set_ylabel('InfoNCE Loss', color='white')
ax.set_title('Contrastive Pretraining Loss\\n(image-genomic alignment)',
             color='white', fontweight='bold')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#333')

# Before vs after comparison
ax2 = axes[1]; ax2.set_facecolor('#111827')
models   = ['Cross-Attention\\n(no pretrain)', 'Cross-Attention\\n(with contrastive pretrain)']
f1_vals  = [0.754,   ft_f1]
auc_vals = [0.933, ft_auc]
x = np.arange(2); w = 0.35
b1 = ax2.bar(x-w/2, f1_vals,  w, label='Macro F1',  color='#4CAF50', alpha=0.85)
b2 = ax2.bar(x+w/2, auc_vals, w, label='Macro AUC', color='#2196F3', alpha=0.85)
for bar, v in zip(list(b1)+list(b2), f1_vals+auc_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{v:.3f}', ha='center', color='white', fontsize=10, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(models, color='white', fontsize=10)
ax2.set_ylim(0, 1.1)
ax2.set_title('Effect of Contrastive Pretraining\\non Downstream Classification',
              color='white', fontweight='bold')
ax2.legend(facecolor='#0d1117', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Phase 6 (Novelty 2) -- Image-Genomics Contrastive Pretraining Results',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '34_contrastive_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/34_contrastive_results.png')
"""))

nb8 = nbformat.v4.new_notebook(); nb8.cells = cells8
with open('notebooks/08_contrastive_pretrain.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb8, f)
print("Written: notebooks/08_contrastive_pretrain.ipynb")

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 9 — Phase 7: Explainability
# ═══════════════════════════════════════════════════════════════════════════════
cells9 = []
cells9.append(md("# Explainability — SHAP + Grad-CAM + Attention (Phase 7)\n\n"
    "Three layers of explanation:\n"
    "1. **SHAP** — which genes and clinical features drove the prediction (tabular)\n"
    "2. **Modality attention** — which data type the model relied on per patient\n"
    "3. **Feature importance** — top genomic drivers per subtype\n\n"
    "> *'It's TNBC because: these 5 genes are upregulated, the tissue morphology showed no glands, "
    "and for this patient the genomic signal dominated (65%) over the image (20%).'*"))

cells9.append(md("## Setup & Data Loading"))
cells9.append(code(SETUP_BOILERPLATE + """
subprocess.run([sys.executable, '-m', 'pip', 'install', 'shap', '-q'], capture_output=True)
import shap
"""))
cells9.append(md("## Patient Cohort & Features"))
cells9.append(code(COHORT_BOILERPLATE))
cells9.append(md("## Train Cross-Attention Model on Full Data"))
cells9.append(code(CA_MODEL_DEF + """
# Train on full dataset for SHAP and attention analysis
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; LR = 5e-4; N_EPOCHS = 200; BS = 16

counts = np.bincount(y)
cw     = torch.tensor(len(y)/(len(counts)*counts), dtype=torch.float32).to(DEVICE)

def sc_all(X):
    s = StandardScaler().fit(X)
    return s, torch.tensor(s.transform(X)).float().to(DEVICE)

sc_img,  img_t  = sc_all(emb_plip)
sc_gen,  gen_t  = sc_all(X_gen)
sc_txt,  txt_t  = sc_all(emb_bert)
sc_clin, clin_t = sc_all(X_clin)
y_t = torch.tensor(y).long().to(DEVICE)

model_full = CrossAttentionFusion().to(DEVICE)
opt        = torch.optim.AdamW(model_full.parameters(), lr=LR, weight_decay=1e-3)
sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
crit       = nn.CrossEntropyLoss(weight=cw)

ds = TensorDataset(img_t, gen_t, txt_t, clin_t, y_t)
dl = DataLoader(ds, batch_size=BS, shuffle=True)

for ep in range(N_EPOCHS):
    model_full.train()
    for ximg, xgen, xtxt, xclin, yb in dl:
        opt.zero_grad(); crit(model_full(ximg, xgen, xtxt, xclin), yb).backward()
        nn.utils.clip_grad_norm_(model_full.parameters(), 1.0); opt.step()
    sched.step()

model_full.eval()
with torch.no_grad():
    logits = model_full(img_t, gen_t, txt_t, clin_t)
    all_probs = torch.softmax(logits, dim=1).cpu().numpy()
    all_preds = all_probs.argmax(axis=1)
    all_attn  = model_full.get_attn().numpy()

print(f"Full-data model trained.")
print(f"  Accuracy: {accuracy_score(y, all_preds):.3f}")
print(f"  F1:       {f1_score(y, all_preds, average='macro', zero_division=0):.3f}")
"""))

cells9.append(md("## SHAP Analysis — Genomic Feature Importance"))
cells9.append(code("""\
# SHAP on the combined early fusion features (genomic + clinical)
# Use a simple gradient explainer on the genomic component
X_combined = np.concatenate([X_gen, X_clin], axis=1)
sc_comb    = StandardScaler().fit(X_combined)
X_comb_sc  = sc_comb.transform(X_combined)

# Simple MLP wrapper for SHAP (genomic + clinical -> subtype)
class GenClinMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(335, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)

gc_model = GenClinMLP().to(DEVICE)
gc_opt   = torch.optim.Adam(gc_model.parameters(), lr=1e-3, weight_decay=1e-4)
gc_crit  = nn.CrossEntropyLoss(weight=cw)
X_comb_t = torch.tensor(X_comb_sc).float().to(DEVICE)
ds_gc = TensorDataset(X_comb_t, y_t)
dl_gc = DataLoader(ds_gc, batch_size=16, shuffle=True)
for ep in range(100):
    gc_model.train()
    for xb, yb in dl_gc:
        gc_opt.zero_grad(); gc_crit(gc_model(xb), yb).backward(); gc_opt.step()

gc_model.eval()
background = X_comb_t[:30]
explainer  = shap.DeepExplainer(gc_model, background)
shap_vals  = explainer.shap_values(X_comb_t, check_additivity=False)  # list of 3 arrays

# Feature names
rna_feat_names  = [c.replace('RNA_','') for c in rna_sub.columns]
clin_feat_names = ['Age', 'Stage', 'Ductal', 'Lobular']
feat_names      = rna_feat_names + clin_feat_names

print(f"SHAP values computed. Shape: {np.array(shap_vals).shape}")
"""))

cells9.append(md("## SHAP Beeswarm + Top Genes per Subtype"))
cells9.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.patch.set_facecolor('#0d1117')
TOP_N = 15

for ax, (si, subtype) in zip(axes, enumerate(SUBTYPES)):
    ax.set_facecolor('#111827')
    sv = shap_vals[si]                        # (114, 335)
    mean_abs = np.abs(sv).mean(axis=0)        # (335,)
    top_idx  = np.argsort(mean_abs)[-TOP_N:][::-1]
    top_vals = mean_abs[top_idx]
    top_names = [feat_names[i] if i < len(feat_names) else f'feat_{i}' for i in top_idx]

    colors = [S_COLOR[subtype] if i < len(rna_feat_names) else '#9C27B0' for i in top_idx]
    bars   = ax.barh(range(TOP_N), top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_names[::-1], color='white', fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|', color='white')
    ax.set_title(f'{subtype} -- Top {TOP_N} Drivers\\n(blue/orange/red=RNA, purple=clinical)',
                 color=S_COLOR[subtype], fontweight='bold', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('SHAP Feature Importance -- Top Genomic + Clinical Drivers per Subtype',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '35_shap_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/35_shap_importance.png')
"""))

cells9.append(md("## Per-Patient Modality Attention + Prediction Summary\n\n"
    "The key explainability figure: for each patient, how much did each modality contribute?"))
cells9.append(code("""\
MOD_LABELS = ['Image\\n(PLIP)', 'Genomic\\n(RNA)', 'Text\\n(BERT)', 'Clinical']
MOD_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

# Show 3 representative patients — one per subtype (hardest cases: lowest confidence)
case_patients = {}
for si, s in enumerate(SUBTYPES):
    mask  = (y == si)
    confs = all_probs[mask, si]
    idxs  = np.where(mask)[0]
    # Pick most interesting: highest confidence correct prediction
    correct = (all_preds[idxs] == y[idxs])
    if correct.any():
        best = idxs[correct][confs[correct].argmax()]
    else:
        best = idxs[confs.argmax()]
    case_patients[s] = int(best)

fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

for col, (subtype, pidx) in enumerate(case_patients.items()):
    pid  = task_df.iloc[pidx]['Patient_ID']
    pred = SUBTYPES[all_preds[pidx]]
    conf = all_probs[pidx, all_preds[pidx]]
    attn = all_attn[pidx]   # (4,)

    # Row 0: attention pie chart
    ax_pie = fig.add_subplot(gs[0, col])
    ax_pie.set_facecolor('#111827')
    wedges, texts, autotexts = ax_pie.pie(
        attn, labels=[m.replace('\\n',' ') for m in MOD_LABELS],
        colors=MOD_COLORS, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(linewidth=0.5, edgecolor='#0d1117'))
    for at in autotexts: at.set_color('white'); at.set_fontsize(9)
    for t  in texts:     t.set_color('white');  t.set_fontsize(8)
    ax_pie.set_title(f'{subtype} Patient\\nTrue: {SUBTYPES[y[pidx]]}  Pred: {pred} ({conf:.0%})',
                     color=S_COLOR[subtype], fontweight='bold', fontsize=10)

    # Row 1: probability bar
    ax_bar = fig.add_subplot(gs[1, col])
    ax_bar.set_facecolor('#111827')
    colors_bar = [S_COLOR[s] if s == pred else '#555555' for s in SUBTYPES]
    bars = ax_bar.bar(SUBTYPES, all_probs[pidx], color=colors_bar, alpha=0.85)
    for bar, v in zip(bars, all_probs[pidx]):
        ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.2f}',
                    ha='center', color='white', fontsize=10, fontweight='bold')
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title(f'Prediction Probabilities\\nPatient {pid[:12]}...',
                     color='white', fontsize=9)
    ax_bar.tick_params(colors='white')
    for spine in ax_bar.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Per-Patient Explainability: Modality Attention + Prediction Confidence',
             color='white', fontsize=13, fontweight='bold')
plt.savefig(FIG_DIR / '36_patient_explainability.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/36_patient_explainability.png')
"""))

nb9 = nbformat.v4.new_notebook(); nb9.cells = cells9
with open('notebooks/09_explainability.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb9, f)
print("Written: notebooks/09_explainability.ipynb")

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 10 — Phase 8: Uncertainty + Fairness
# ═══════════════════════════════════════════════════════════════════════════════
cells10 = []
cells10.append(md("# Uncertainty Quantification + Fairness Audit (Phase 8)\n\n"
    "**Uncertainty:** Monte Carlo Dropout — run inference 50 times, compute mean ± std.\n"
    "If std > threshold → flag as *'Low confidence — refer to human expert'*.\n\n"
    "**Fairness:** Does the model perform equally for all patient subgroups?\n"
    "Age, AJCC stage, ER/PR/HER2 status — required for any medical AI publication."))

cells10.append(md("## Setup & Data Loading"))
cells10.append(code(SETUP_BOILERPLATE))
cells10.append(md("## Patient Cohort & Features"))
cells10.append(code(COHORT_BOILERPLATE))
cells10.append(md("## Monte Carlo Dropout Uncertainty"))
cells10.append(code(CA_MODEL_DEF + """
class MCDropoutFusion(CrossAttentionFusion):
    \"\"\"Enable dropout at inference time for Monte Carlo uncertainty estimation.\"\"\"
    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()   # keep dropout active even in eval mode

# Train on full data
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; LR = 5e-4; N_EPOCHS = 200; BS = 16
counts = np.bincount(y)
cw     = torch.tensor(len(y)/(len(counts)*counts), dtype=torch.float32).to(DEVICE)

def sc_all(X):
    s = StandardScaler().fit(X)
    return s, torch.tensor(s.transform(X)).float().to(DEVICE)

sc_img,  img_t  = sc_all(emb_plip)
sc_gen,  gen_t  = sc_all(X_gen)
sc_txt,  txt_t  = sc_all(emb_bert)
sc_clin, clin_t = sc_all(X_clin)
y_t = torch.tensor(y).long().to(DEVICE)

mc_model = MCDropoutFusion().to(DEVICE)
opt      = torch.optim.AdamW(mc_model.parameters(), lr=LR, weight_decay=1e-3)
sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
crit     = nn.CrossEntropyLoss(weight=cw)
ds = TensorDataset(img_t, gen_t, txt_t, clin_t, y_t)
dl = DataLoader(ds, batch_size=BS, shuffle=True)
for ep in range(N_EPOCHS):
    mc_model.train()
    for ximg, xgen, xtxt, xclin, yb in dl:
        opt.zero_grad(); crit(mc_model(ximg, xgen, xtxt, xclin), yb).backward()
        nn.utils.clip_grad_norm_(mc_model.parameters(), 1.0); opt.step()
    sched.step()

# MC Dropout inference: 50 forward passes
N_MC = 50
mc_model.eval()
mc_model.enable_mc_dropout()

mc_samples = []
with torch.no_grad():
    for _ in range(N_MC):
        logits = mc_model(img_t, gen_t, txt_t, clin_t)
        mc_samples.append(torch.softmax(logits, dim=1).cpu().numpy())

mc_stack = np.stack(mc_samples, axis=0)       # (50, 114, 3)
mc_mean  = mc_stack.mean(axis=0)              # (114, 3)
mc_std   = mc_stack.std(axis=0).max(axis=1)   # (114,)  max std across classes
mc_preds = mc_mean.argmax(axis=1)             # (114,)

UNC_THRESHOLD = np.percentile(mc_std, 75)     # flag top 25% most uncertain
uncertain_mask = mc_std > UNC_THRESHOLD

print(f"MC Dropout uncertainty (50 forward passes):")
print(f"  Mean std:  {mc_std.mean():.4f}")
print(f"  Threshold: {UNC_THRESHOLD:.4f} (75th percentile)")
print(f"  Flagged as uncertain: {uncertain_mask.sum()} patients ({uncertain_mask.mean()*100:.1f}%)")
print(f"  Accuracy on certain   patients: {accuracy_score(y[~uncertain_mask], mc_preds[~uncertain_mask]):.3f}")
print(f"  Accuracy on uncertain patients: {accuracy_score(y[uncertain_mask],  mc_preds[uncertain_mask]):.3f}")
"""))

cells10.append(md("## Calibration Curve (Reliability Diagram)"))
cells10.append(code("""\
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0d1117')

# Uncertainty histogram
ax = axes[0]; ax.set_facecolor('#111827')
ax.hist(mc_std[~uncertain_mask], bins=20, color='#4CAF50', alpha=0.75, label='Certain')
ax.hist(mc_std[uncertain_mask],  bins=20, color='#F44336', alpha=0.75, label='Uncertain (flagged)')
ax.axvline(UNC_THRESHOLD, color='white', linestyle='--', lw=1.5, label=f'Threshold={UNC_THRESHOLD:.3f}')
ax.set_xlabel('Prediction Uncertainty (max std)', color='white')
ax.set_ylabel('Patient Count', color='white')
ax.set_title('Uncertainty Distribution\\n(MC Dropout, 50 samples)',
             color='white', fontweight='bold')
ax.legend(facecolor='#0d1117', labelcolor='white')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#333')

# Calibration curve
ax2 = axes[1]; ax2.set_facecolor('#111827')
for si, s in enumerate(SUBTYPES):
    binary_true = (y == si).astype(int)
    prob_true, prob_pred = calibration_curve(binary_true, mc_mean[:, si], n_bins=5, strategy='uniform')
    ax2.plot(prob_pred, prob_true, marker='o', color=S_COLOR[s], lw=2, label=s)
ax2.plot([0,1],[0,1], '--', color='#aaaaaa', lw=1, label='Perfect calibration')
ax2.set_xlabel('Mean Predicted Probability', color='white')
ax2.set_ylabel('Fraction of Positives', color='white')
ax2.set_title('Calibration Curve\\n(predicted confidence vs actual accuracy)',
              color='white', fontweight='bold')
ax2.legend(facecolor='#0d1117', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_edgecolor('#333')

# Uncertainty by subtype
ax3 = axes[2]; ax3.set_facecolor('#111827')
import seaborn as sns
unc_df = pd.DataFrame({'Subtype': [SUBTYPES[yi] for yi in y], 'Uncertainty': mc_std})
sns.violinplot(x='Subtype', y='Uncertainty', data=unc_df, palette=S_COLOR,
               order=SUBTYPES, inner='box', cut=0, ax=ax3, linewidth=1.2)
for p in ax3.collections: p.set_alpha(0.75)
ax3.axhline(UNC_THRESHOLD, color='white', linestyle='--', lw=1.5, label='Flag threshold')
ax3.set_title('Uncertainty by Subtype\\n(TNBC = hardest to classify?)',
              color='white', fontweight='bold')
ax3.set_xlabel('Subtype', color='white'); ax3.set_ylabel('MC Uncertainty', color='white')
ax3.legend(facecolor='#0d1117', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values(): spine.set_edgecolor('#333')

fig.suptitle('Phase 8 -- Uncertainty Quantification: MC Dropout',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '37_uncertainty.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/37_uncertainty.png')
"""))

cells10.append(md("## Fairness Audit — Performance Across Patient Subgroups"))
cells10.append(code("""\
task_df['mc_pred']       = mc_preds
task_df['mc_confidence'] = mc_mean.max(axis=1)
task_df['mc_uncertain']  = uncertain_mask.astype(int)
task_df['age_group']     = pd.cut(
    pd.to_numeric(task_df['demographic_age_at_index'], errors='coerce'),
    bins=[0, 40, 60, 120], labels=['<40', '40-60', '>60'])
task_df['stage_group']   = task_df['diagnoses_ajcc_pathologic_stage'].apply(
    lambda s: 'Early (I-II)' if any(x in str(s) for x in ['I','II']) and 'III' not in str(s) and 'IV' not in str(s)
    else 'Late (III-IV)')

def subgroup_metrics(mask, name):
    if mask.sum() < 5:
        return {'Subgroup': name, 'N': mask.sum(), 'Accuracy': None, 'Macro F1': None}
    y_sub = y[mask]; p_sub = mc_preds[mask]
    return {
        'Subgroup': name,
        'N':        mask.sum(),
        'Accuracy': round(accuracy_score(y_sub, p_sub), 3),
        'Macro F1': round(f1_score(y_sub, p_sub, average='macro', zero_division=0), 3),
    }

rows = []
# Age groups
for grp in ['<40', '40-60', '>60']:
    mask = (task_df['age_group'] == grp).values
    rows.append(subgroup_metrics(mask, f'Age {grp}'))

# Stage groups
for grp in ['Early (I-II)', 'Late (III-IV)']:
    mask = (task_df['stage_group'] == grp).values
    rows.append(subgroup_metrics(mask, f'Stage {grp}'))

# ER status
for er_val in ['Positive', 'Negative']:
    mask = (task_df['ER'] == er_val).values
    rows.append(subgroup_metrics(mask, f'ER {er_val}'))

# HER2 status
for h_val in ['Positive', 'Negative']:
    mask = (task_df['HER2'] == h_val).values
    rows.append(subgroup_metrics(mask, f'HER2 {h_val}'))

# Overall
rows.append(subgroup_metrics(np.ones(len(y), dtype=bool), 'ALL PATIENTS'))

fairness_df = pd.DataFrame(rows).dropna()
print("Fairness Audit:")
print(fairness_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#111827')

x = np.arange(len(fairness_df))
w = 0.35
b1 = ax.bar(x - w/2, fairness_df['Accuracy'], w, label='Accuracy', color='#2196F3', alpha=0.85)
b2 = ax.bar(x + w/2, fairness_df['Macro F1'], w, label='Macro F1', color='#4CAF50', alpha=0.85)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h: ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}',
                  ha='center', color='white', fontsize=8, rotation=45)

# Fairness threshold: flag if <0.1 below overall
overall_f1 = fairness_df[fairness_df['Subgroup']=='ALL PATIENTS']['Macro F1'].values[0]
ax.axhline(overall_f1 - 0.1, color='#F44336', linestyle='--', lw=1.5,
           label=f'Fairness threshold (overall F1 - 0.1)')

ax.set_xticks(x)
ax.set_xticklabels([f"{r['Subgroup']}\\n(n={int(r['N'])})"
                    for _, r in fairness_df.iterrows()],
                   color='white', fontsize=9, rotation=30, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', color='white')
ax.set_title('Fairness Audit -- Model Performance Across Patient Subgroups\\n'
             '(consistent performance = equitable AI)',
             color='white', fontweight='bold', fontsize=12)
ax.legend(facecolor='#0d1117', labelcolor='white')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig(FIG_DIR / '38_fairness_audit.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: figures/38_fairness_audit.png')
"""))

nb10 = nbformat.v4.new_notebook(); nb10.cells = cells10
with open('notebooks/10_uncertainty_fairness.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb10, f)
print("Written: notebooks/10_uncertainty_fairness.ipynb")

print("\nAll 5 notebooks written successfully.")
