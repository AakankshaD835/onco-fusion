# Multi-Modal Fusion of Histopathology Images, Genomics, and Clinical Data for Breast Cancer Subtype Classification and Survival Prediction

**Project Type:** Master's Thesis / Research Project  
**Domain:** Computational Oncology · Deep Learning · Precision Medicine  
**Stack:** PyTorch · HuggingFace Transformers · Groq API · py3Dmol · NetworkX · SHAP  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Motivation](#2-motivation)
3. [Datasets](#3-datasets)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Models and Components](#5-models-and-components)
6. [Build Plan — Phase by Phase](#6-build-plan--phase-by-phase)
7. [Novelty Contributions](#7-novelty-contributions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Folder Structure](#9-folder-structure)
10. [Dependencies](#10-dependencies)

---

## 1. Problem Statement

Breast cancer is the most commonly diagnosed cancer in women worldwide. Accurate classification of cancer subtypes — HR+ (Luminal A + Luminal B combined), HER2-enriched, and Triple-Negative Breast Cancer (TNBC) — together with reliable survival prediction, is critical for treatment planning.

> **Implementation note (Phase 1):** The TCGA-BRCA dataset does not contain an explicit subtype column. Subtypes were derived from ER/PR/HER2 receptor status in `Clinical_Treatment_Data.csv`: HR+ = ER+ or PR+ (HER2-), HER2+ = HER2+, TNBC = ER−, PR−, HER2−. LumA and LumB were merged into HR+ due to small per-subclass sample sizes (n=114 task cohort). 7 patients with unknown receptor status were excluded.

Current clinical practice analyses data in silos: a pathologist reads the histopathology slide, a genomicist analyses gene expression, and a clinician reviews the patient record independently. Existing machine learning approaches mirror this limitation by operating on a single data modality. The few multi-modal methods that exist suffer from three compounding problems:

1. They use only two modalities (typically image + clinical), discarding genomic signal.
2. They assume all modalities are present at test time, which is clinically unrealistic — genomic sequencing is expensive and often unavailable.
3. They produce a prediction label with no interpretable reasoning, making clinical adoption impossible.

This project addresses all three limitations and extends the pipeline to include drug target identification and treatment suggestion, producing a complete precision medicine system.

---

## 2. Motivation

A complete precision medicine pipeline for breast cancer must answer three clinical questions from a single patient encounter:

| Clinical Question | Why It Matters |
|---|---|
| What subtype is this tumour? | Determines the chemotherapy protocol (e.g., HER2+ → Herceptin, TNBC → aggressive chemo) |
| What is this patient's survival risk? | Guides the aggressiveness and duration of treatment |
| Which drugs is this patient likely to respond to? | Enables targeted therapy and drug repurposing decisions |

This project builds one unified model that answers all three simultaneously, uses four data modalities derived from two Kaggle datasets, and degrades gracefully — rather than failing — when modalities are missing.

---

## 3. Datasets

This project uses exactly **two datasets**, both from Kaggle.

---

### Dataset 1 — TCGA-BRCA Multi-Modal Fusion Dataset

| Attribute | Detail |
|---|---|
| Source | Kaggle |
| Dataset ID | `sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready` |
| Kaggle Download | `kaggle datasets download sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready` |
| kagglehub | `kagglehub.dataset_download('sepehreslamimoghadam/breast-cancer-vision-and-genomic-fusion-ml-ready')` |
| Size | ~10 GB |
| Cohort | TCGA-BRCA (The Cancer Genome Atlas — Breast Invasive Carcinoma) |

**What it contains — all three modalities for the same patients, pre-aligned:**

| Modality | Content |
|---|---|
| Histopathology Images | Whole-slide SVS patches from TCGA-BRCA patients (107/114 task patients have patches) |
| RNA-seq Gene Expression | 59,427 genes in `RNA_RAW` (deduplicated to 122 unique patients); 300 RNA + 31 CNV features in `RNA_CNV_ModelReady` (331 total) |
| Clinical Data | Age, AJCC tumour stage, histological type, ER/PR/HER2 receptor status, overall survival (days), vital status, treatment history |

**Actual aligned cohort:** 122 patients total → 121 with all 5 modalities → **114 task cohort** (7 excluded: unknown receptor status).  
**Class distribution:** HR+ 85 (74.6%), HER2+ 18 (15.8%), TNBC 11 (9.6%) — heavily imbalanced.

**Confirmed clinical files:**

```
Clinical_Treatment_Data.csv   ← ER/PR/HER2 receptor status, histological type, demographics
Clinical_Patient_Data.csv     ← OS days, vital status, AJCC stage
RNA_CNV_ModelReady.csv        ← 300 RNA + 31 CNV features (pre-selected, model-ready)
RNA_RAW.csv                   ← 59,427 gene expression values (deduplicated)
Mutations_Dataset.csv         ← somatic mutation data
```

> **Note:** No explicit SUBTYPE column exists. Subtypes derived from ER/PR/HER2 IHC columns in `Clinical_Treatment_Data.csv` during Phase 1. No grade column present — histological type (Ductal/Lobular) used as proxy.

---

### Dataset 2 — Genomics of Drug Sensitivity in Cancer (GDSC)

| Attribute | Detail |
|---|---|
| Source | Kaggle |
| Dataset ID | `samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc` |
| Kaggle Download | `kaggle datasets download samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc` |
| kagglehub | `kagglehub.dataset_download('samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc')` |
| Size | ~75 MB |

**What it contains:**

| Content | Detail |
|---|---|
| Drug response (IC50) | Measured drug sensitivity values (ln IC50) for 250+ drugs across 1,000+ cancer cell lines |
| Genomic features | Gene expression profiles of the cancer cell lines |
| Cell line metadata | Tissue type (including breast), cancer subtype, mutation status |

**How it connects to Dataset 1:**  
Gene expression from TCGA-BRCA patients (Dataset 1) is correlated with drug sensitivity patterns in matched breast cancer cell lines (Dataset 2). This enables the drug repurposing pipeline in Phase 10: *"Patients with high PIK3CA expression are likely resistant to Drug X".*

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATASET 1 — TCGA-BRCA                           │
│                                                                     │
│   [Histopathology Patch]   [Clinical CSV]   [RNA-seq Expression]    │
└──────────┬──────────────────────┬───────────────────┬──────────────┘
           │                      │                   │
           ▼                      ▼                   ▼
     BiomedCLIP               TabNet              1D-CNN
     (512-d embed)            (128-d embed)       (256-d embed)
           │                      │                   │
           │        [Synthetic Text Report]            │
           │         (generated from Clinical)         │
           │                  ▼                        │
           │          BioClinicalBERT                  │
           │           (256-d embed)                   │
           │                  │                        │
           └──────────────────┴────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │     Cross-Attention Fusion       │
               │     Transformer Encoder          │
               │  [img | clin | gen | text tokens]│
               └────────────────┬────────────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
   Subtype Head           Survival Head           Grade Head
   (4-class softmax)      (Cox loss)              (3-class softmax)
   LumA / LumB /          Risk score →            Grade 1 / 2 / 3
   HER2+ / TNBC           C-index metric


                    ┌──────────────────────────────┐
                    │   DATASET 2 — GDSC            │
                    │   Drug sensitivity (IC50)     │
                    │   + Phase 10 drug repurposing │
                    └──────────────────────────────┘
```

---

## 5. Models and Components

### 5.1 Pretrained Encoders — Loaded from HuggingFace (No Local Download Needed)

| Model | HuggingFace ID | Modality | Role |
|---|---|---|---|
| **PLIP** ✓ | `vinid/plip` | Image | **Selected encoder** — best Phase 2 AUC (0.743) |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Image | Comparison (Phase 2 AUC 0.653) |
| Phikon | `owkin/phikon` | Image | Comparison (Phase 2 AUC 0.666) |
| BioClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | Text | Encodes synthetic clinical text reports (Phase 2 AUC 0.562) |

All models are loaded via `transformers` or `open_clip_torch` at runtime — no manual file management required.

### 5.2 Custom PyTorch Components — Built From Scratch

| Component | Input | Architecture | Output |
|---|---|---|---|
| 1D-CNN Genomic Encoder ✓ | 331 features (300 RNA + 31 CNV) | Conv1d(1→64→128→256) + AdaptiveAvgPool + Dropout | 256-d embedding |
| Clinical MLP Encoder | 4 features (age, stage, ductal, lobular) — **ER/PR/HER2 excluded (label leakage)** | Linear(4→64→256) + ReLU | 256-d embedding |
| Cross-Attention Fusion Module | 4 modality tokens (img, clin, gen, text) | Multi-head cross-attention transformer | Fused representation |
| Subtype Classification Head | Fused embedding | Linear → softmax | **3 class probabilities** (HR+, HER2+, TNBC) |
| Survival Prediction Head | Fused embedding | Linear → scalar + Cox loss | Continuous risk score |
| Grade Classification Head | Fused embedding | Linear → softmax | 3 class probabilities |

### 5.3 LLM for Clinical Report Generation

| Attribute | Detail |
|---|---|
| Provider | Groq API |
| Model | Llama 3.3 70B |
| Cost | Free — 14,400 requests/day on the free tier |
| Setup | groq.com → create free account → copy API key → add to `.env` |
| Purpose | Generates a structured 3-sentence clinical summary from model predictions + SHAP feature attributions, delivered to the attending physician |

---

## 6. Build Plan — Phase by Phase

### Phase 1 — Setup and Exploratory Data Analysis

**Goal:** Fully understand all data before building any model.

**Image modality:**
- Download Dataset 1 via `kagglehub`
- Locate and load histopathology patches
- Visualise sample patches per subtype

**Clinical modality:**
- Load `clinical.csv` from Dataset 1
- Plot age distribution, AJCC stage breakdown, ER/PR/HER2 receptor status, subtype distribution
- Confirm `SUBTYPE` column: LumA / LumB / HER2+ / TNBC
- Check missing values, data types, outliers

**Genomic modality:**
- Load RNA-seq expression CSV from Dataset 1
- Plot gene expression heatmap (top 20 most variable genes across 50 patients)
- Compute variance across genes — identify top variable features

**Survival analysis:**
- Plot Kaplan-Meier overall survival curve (OS_MONTHS + OS_STATUS)
- Stratify KM curves by subtype and by grade

**Drug sensitivity (Dataset 2):**
- Load GDSC IC50 data
- Explore tissue type distribution — extract breast cancer cell lines
- Plot top 15 most tested drugs, IC50 value distributions

**Output:** `notebooks/01_EDA.ipynb`

---

### Phase 2 — Unimodal Baselines

**Goal:** Establish what each modality achieves in isolation. Every fusion paper must demonstrate that fusion outperforms any single modality — this phase provides that baseline.

**Image alone:**
- Resize patches to 224×224
- Extract embeddings: BiomedCLIP vs Phikon vs PLIP (compare all three)
- Train a linear probe on each set of embeddings
- Report: Accuracy, AUC (macro OvR), F1 (macro)
- Select the best image encoder → carry forward into fusion

**Clinical tabular alone:**
- Features: age, stage, ductal/lobular histology — **ER/PR/HER2 excluded** (they define subtype labels → data leakage)
- Encoders: TabNet, XGBoost, SVM
- Report same metrics

**Genomic alone:**
- 59,427 genes → PCA(64) → XGBoost (PCA limited to 64 due to n=91 training samples per fold)
- 331 features (RNA_CNV_ModelReady) → 1D-CNN end-to-end
- Compare both approaches

**Synthetic text alone:**
- Construct a free-text report per patient: *"Female patient, 41 years old. Infiltrating Ductal Carcinoma, Stage I breast cancer."*
- **ER/PR/HER2 excluded** from text — they define subtype labels
- Encode with BioClinicalBERT → linear classifier

**Output:** `notebooks/02_unimodal_baselines.ipynb` ✓ **COMPLETE**

**Phase 2 Results (5-fold stratified CV, macro-averaged, class_weight=balanced):**

| Modality | Model | Macro F1 | AUC |
|---|---|---|---|
| Image | **PLIP** ← selected | **0.450** | **0.743** |
| Image | BiomedCLIP | 0.366 | 0.653 |
| Image | Phikon | 0.366 | 0.666 |
| Clinical | SVM ← selected | 0.313 | 0.388 |
| Clinical | XGBoost | 0.278 | 0.504 |
| Clinical | TabNet | 0.228 | 0.431 |
| Genomic | **1D-CNN** ← selected | **0.502** | **0.833** |
| Genomic | PCA+XGBoost | 0.483 | 0.821 |
| Text | **BioClinicalBERT** ← selected | **0.366** | **0.562** |

Best unimodal: Genomic 1D-CNN (F1=0.502, AUC=0.833). Fusion target: significantly exceed all baselines.

---

### Phase 3 — Fusion Models (3 notebooks)

**Goal:** Show that combining modalities outperforms any single-modality baseline. Three fusion strategies compared in order of complexity.

---

#### Phase 3a — Early Fusion (`notebooks/03_early_fusion.ipynb`) ✓ COMPLETE

Concatenate all 4 modality embeddings → single MLP classifier. Simplest possible fusion.

```
[PLIP(512) || BioClinicalBERT(768) || Genomic(331) || Clinical(4)] = 1615-d
        → EarlyFusionMLP: 1615 → 512 → 256 → 64 → 3
        → HR+ / HER2+ / TNBC
        (BatchNorm + Dropout(0.4), class_weight=balanced, 5-fold stratified CV)
```

**Phase 3a Results (5-fold stratified CV, macro-averaged):**

| Fold | Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| 1 | 0.826 | 0.756 | 0.847 |
| 2 | 0.870 | 0.794 | 0.881 |
| 3 | 0.783 | 0.741 | 0.918 |
| 4 | 0.783 | 0.511 | 0.817 |
| 5 | 0.909 | 0.869 | 0.908 |
| **Mean** | **0.834 ± 0.055** | **0.734 ± 0.134** | **0.874 ± 0.042** |

**Lift over best unimodal (Genomic 1D-CNN, F1=0.502, AUC=0.833): F1 +0.232 | AUC +0.041**

**Figures generated:** `10_he_patch_gallery.png`, `11_genomic_heatmap.png`, `12_clinical_profiles.png`, `13_clinical_text_reports.png`, `14_early_fusion_confusion.png`, `15_early_fusion_roc.png`, `16_patient_case_studies.png`, `17_km_early_fusion.png`, `18_early_fusion_comparison.png`

**Visualisations (medical-AI style):**
- H&E stained tissue patch gallery per subtype (4 patches × 3 subtypes)
- Top-30 differentially expressed gene heatmap (patients × genes, z-scored)
- Clinical profile distributions per subtype (age, stage, histological type)
- Confusion matrix labelled HR+/HER2+/TNBC with clinical misclassification annotation
- Per-subtype ROC curves (one-vs-rest) with AUC per subtype + FPR=0.1 clinical threshold
- Patient case studies: all 4 modalities + prediction confidence per patient
- Kaplan-Meier curves by true vs predicted subtype labels
- Bar chart: early fusion vs each unimodal baseline (F1 + AUC)

---

#### Phase 3b — Late Fusion (`notebooks/04_late_fusion.ipynb`)

Train 4 independent modality classifiers → weighted ensemble of output probabilities.

```
PLIP → Classifier_img → P(subtype | image)
1D-CNN → Classifier_gen → P(subtype | genomic)     → weighted average → final prediction
BERT → Classifier_text → P(subtype | text)
Clinical → Classifier_clin → P(subtype | tabular)
```

**Visualisations (medical-AI style):**
- Ensemble weight optimisation plot (which modality contributes most)
- Per-subtype ROC curves — late fusion vs early fusion vs best unimodal
- Misclassification table: which patients flipped from wrong → correct vs correct → wrong
- Stacked bar: modality contribution per subtype (HR+ relies on what? TNBC on what?)

---

#### Phase 3c — Cross-Attention Fusion (`notebooks/05_cross_attention_fusion.ipynb`) — MAIN MODEL

Each modality projected to 256-d token → multi-head cross-attention transformer → 3 task heads.

```python
tokens = stack([proj_img(plip), proj_gen(gen_feats), proj_text(bert), proj_clin(clin)])
# shape: (batch, 4, 256)
fused = TransformerEncoder(tokens)   # cross-attention across modalities
subtype_logits = subtype_head(fused.mean(1))
survival_score = survival_head(fused.mean(1))
grade_logits   = grade_head(fused.mean(1))
```

**Visualisations (medical-AI style):**
- Confusion matrix with clinical interpretation (which subtype pairs are confused)
- Per-subtype ROC + AUC — comparing all 3 fusion strategies + best unimodal
- Modality attention weights heatmap per patient (114 × 4 heatmap)
- Attention weight bar chart per subtype: *"For TNBC patients, genomic contributes X%, image Y%"*
- Kaplan-Meier curves stratified by predicted subtype (predicted HR+ vs HER2+ vs TNBC)
- Calibration curve: predicted confidence vs actual accuracy
- Summary comparison table: unimodal → early → late → cross-attention (F1, AUC, Acc)

**Output:** 3 notebooks complete  
**Deliverable:** Cross-attention beats early fusion beats late fusion beats best unimodal — each step justified clinically.

---

### Phase 4 — Multi-Task Learning (`notebooks/06_multitask.ipynb`)

**Goal:** Train one shared encoder to answer three clinical questions simultaneously, improving data efficiency and generalisation.

```
Shared Cross-Attention Fusion Encoder
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 Subtype   Survival    Grade
 (3-class) (Cox loss)  (2-class: Ductal/Lobular proxy)
```

**Combined training loss:**

```python
total_loss = λ1 * cross_entropy(subtype) + λ2 * cox_loss(survival) + λ3 * cross_entropy(grade)
```

The λ weights are tunable hyperparameters that balance task contributions during training.

---

### Phase 5 — Novelty 1: Modality Dropout Training (`notebooks/07_modality_dropout.ipynb`)

**Goal:** Train the model to perform robustly when one or more modalities are missing at test time.

**Motivation:** Real hospitals do not always have complete multi-modal data. Genomic sequencing is expensive, whole-slide images require specialised scanners, and clinical records may be incomplete. Every existing multi-modal paper assumes complete data at test time — this is clinically unrealistic. This phase makes the model deployable in real-world hospital settings.

**Method:** During each training batch, randomly drop a subset of modalities by zeroing their embeddings before the fusion step.

```python
active = random.sample(['image', 'clinical', 'genomic', 'text'], k=random.randint(2, 4))
# Zero-mask embeddings for inactive modalities before cross-attention
```

**Expected results:**

| Modalities Available at Test Time | Macro F1 |
|---|---|
| All 4 modalities | best |
| Missing genomics | slight drop |
| Missing image | slight drop |
| Missing genomics + text | moderate drop |
| Standard model (no dropout training) — any modality missing | collapses |

---

### Phase 6 — Novelty 2: Cross-Modal Contrastive Pretraining (`notebooks/08_contrastive_pretrain.ipynb`)

**Goal:** Before supervised fine-tuning, pre-align the image and genomic encoders in a shared latent space using contrastive learning.

**Method:**

- **Positive pairs:** image embedding + genomic embedding from the *same* TCGA patient
- **Negative pairs:** image + genomic embeddings from *different* patients
- **Loss:** InfoNCE (same loss used in CLIP)
- Pretrain encoders for 20 epochs → then fine-tune on all downstream tasks

**Why this is novel:**  
BiomedCLIP performs image–text contrastive learning. No prior published work has applied image–genomics contrastive pretraining specifically to breast cancer. This is an unexplored modality pairing.

**What is demonstrated:**
- UMAP visualisation: after pretraining, same-patient image and genomic embeddings cluster together in shared space
- Downstream classification accuracy improves by X% compared to no pretraining

**Output:** `notebooks/08_contrastive_pretrain.ipynb`

---

### Phase 7 — Explainability

**Goal:** Provide interpretable, patient-level explanations for every prediction.

**SHAP analysis (clinical + genomic features):**
- Which genes are most predictive of malignant subtype?
- Which clinical features dominate survival predictions?
- Beeswarm plot (global feature importance), waterfall plot (per-patient), force plot (individual prediction breakdown)

**Grad-CAM (image patches):**
- Which tissue regions did the model attend to when classifying this patch?
- Overlay heatmap on the original histopathology image

**Modality attention weights:**
- The cross-attention module assigns a weight to each modality token per patient
- Report: *"For this TNBC patient: 65% image, 25% genomic, 10% clinical"*
- Visualise as a pie chart — one per patient type and one per subtype group

**Key figure for the thesis:**  
For 3 representative patients (one per hard case), show:  
prediction label + confidence + Grad-CAM patch + top-5 SHAP genes + modality attention pie chart

**Output:** `notebooks/09_explainability.ipynb`

---

### Phase 8 — Uncertainty Quantification and Fairness Analysis

**Goal:** Quantify prediction confidence and audit model equity across demographic and clinical subgroups.

**Uncertainty:**
- Monte Carlo Dropout: run inference 50 times with dropout active → compute mean ± std per patient
- Decision rule: if std > threshold → flag prediction as *"Low confidence — refer to human expert"*
- Calibration curve: plot predicted confidence vs actual accuracy (reliability diagram)
- Metric: Expected Calibration Error (ECE)

**Fairness audit:**

Stratify all performance metrics (Accuracy, F1, C-index) by the following patient subgroups:

| Subgroup | Categories |
|---|---|
| Age | < 40, 40–60, > 60 |
| ER/PR/HER2 receptor status | All combinations |
| AJCC cancer stage | Stage I, II, III, IV |
| Tumour grade | Grade 1, 2, 3 |
| Race / ethnicity | White, Black/African American, Other |

The goal is to confirm — or identify failures in — equitable model performance across all patient groups. This analysis is now required in medical AI publications.

**Output:** `notebooks/10_uncertainty_fairness.ipynb`

---

### Phase 9 — Drug Target Identification and Repurposing

**Goal:** Extend the pipeline from diagnosis and prognosis into actionable treatment suggestion using GDSC (Dataset 2). This phase completes the precision medicine loop — from raw patient data all the way to drug recommendation.

**Full pipeline:**

```
RNA-seq gene expression (Dataset 1 — TCGA-BRCA)
        ↓
Differential expression analysis (malignant vs benign subtypes)
        ↓
Identify top upregulated genes → candidate drug targets
        ↓
Cross-reference with GDSC (Dataset 2):
  Does high expression of gene X correlate with drug resistance (high IC50)?
        ↓
Fetch AlphaFold2 protein structure for target gene
  (free, pre-computed for all human proteins — no compute required)
        ↓
Visualise 3D protein structure and binding pockets (py3Dmol — interactive in notebook)
        ↓
DeepPurpose: predict drug-target binding affinity
  Input: protein sequence of target gene
  Output: ranked list of known drugs + predicted binding affinity scores
        ↓
Drug repurposing candidates per patient subtype
```

**Four concrete notebook additions:**

| Addition | Method | Tool |
|---|---|---|
| Differentially expressed genes | Compare mean expression malignant vs benign subtypes | `scipy.stats.ttest_ind` |
| Drug sensitivity correlation | Spearman correlation: gene expression (TCGA) vs IC50 (GDSC) | pandas + scipy |
| Protein structure visualisation | Fetch .pdb from AlphaFold2 by UniProt ID | `py3Dmol` |
| Drug repurposing | Pretrained drug-target interaction model | `DeepPurpose` |

**Project narrative without Phase 10:**
> *"We classify breast cancer subtypes and predict patient survival."*

**Project narrative with Phase 10:**
> *"We classify breast cancer subtypes, predict survival, identify genomic drug targets from expression data, and suggest repurposed drugs for each patient subtype — a complete precision medicine pipeline from diagnosis to treatment recommendation."*

**Output:** `notebooks/09_drug_discovery.ipynb`

---

### Phase 10 — LLM Clinical Report Generation (Final Phase)

**Goal:** Synthesise all results from the entire pipeline — subtype prediction, survival risk, grade, SHAP feature attributions, modality attention weights, uncertainty flag, and drug repurposing candidates — into a single structured clinical report generated by a large language model.

This is the final output of the precision medicine pipeline: a report a clinician can actually read and act on.

**What feeds into the report:**

| Source | Contribution |
|---|---|
| Phase 3c Cross-Attention | Predicted subtype + confidence + modality attention weights |
| Phase 4 Multi-Task | Survival risk score + predicted grade |
| Phase 7 Explainability | Top-5 SHAP genes + Grad-CAM tissue region |
| Phase 8 Uncertainty | Confidence flag — high confidence vs "refer to expert" |
| Phase 9 Drug Discovery | Top drug candidates ranked by predicted binding affinity |

**LLM: Groq API (Llama 3.3 70B — free tier)**

```python
prompt = f"""
You are an oncology AI assistant. Generate a structured clinical report for the attending physician.

Patient: {age}F | Stage {stage} | ER:{er} PR:{pr} HER2:{her2}

Model predictions:
  Subtype       : {subtype} (confidence {conf:.0f}%)
  Survival risk : {risk_pct:.0f}th percentile (high/low risk)
  Grade         : {grade}
  Uncertainty   : {'LOW — high confidence' if conf > 0.75 else 'HIGH — recommend expert review'}

Key genomic drivers (SHAP): {', '.join(top_genes)}
Modality reliance: Image {attn_img:.0f}% | Genomic {attn_gen:.0f}% | Text {attn_txt:.0f}% | Clinical {attn_clin:.0f}%
Drug repurposing candidates: {', '.join(top_drugs)}

Write a 4-sentence structured clinical summary covering:
1. Subtype classification and confidence
2. Survival prognosis
3. Key molecular drivers and treatment implications
4. Recommended next steps
"""
```

**Visualisations:**
- Per-patient full clinical report card (all predictions + explanations in one figure)
- Side-by-side: AI report text + tissue patch (Grad-CAM) + gene bar chart
- Report quality audit: does predicted subtype match known receptor status?

**Output:** `notebooks/10_llm_report.ipynb`

---

## 6b. Visualization Philosophy — Medical-First

All notebooks present results in the visual language of clinical and pathology literature, not generic ML dashboards. Each modality is shown in its natural medical form:

| Modality | Primary Visualization Style |
|---|---|
| **Histopathology (Image)** | H&E stained tissue patch grids per subtype; Grad-CAM attention overlaid directly on tissue (Phase 7) |
| **Genomics** | Gene expression heatmaps (patients × genes); differential expression volcano plots; subtype signature boxplots per key gene (ESR1, ERBB2, PGR) |
| **Clinical Text** | Actual rendered clinical report text per patient; word-level attention highlighting |
| **Tabular Clinical** | Radar charts of clinical parameters per subtype; Kaplan-Meier survival curves (standard oncology format) |
| **Fusion** | Modality attention weight heatmap (patients × 4 modalities); patient case panels showing all 4 modalities + prediction side by side |
| **Drug Discovery (Phase 9)** | 3D protein structure with binding pocket highlighted (py3Dmol); drug-target affinity ranked table |
| **LLM Report (Phase 10)** | Full structured clinical report card per patient — all predictions, explanations, and drug candidates in one readable output |

The goal: a reviewer or clinician who opens any notebook immediately recognises the medical context — tissue morphology, receptor status, survival curves — before seeing any model metrics.

---

## 7. Novelty Contributions

| # | Contribution | What | Why It Matters |
|---|---|---|---|
| N1 | 4-modality fusion | Image + Clinical + Genomic + Synthetic Text | Most published papers use only 2 modalities |
| N2 | Modality dropout training | Model works with incomplete data at test time | Clinically deployable; not just an academic benchmark |
| N3 | Image-genomics contrastive pretraining | New positive pair: same-patient histopathology + RNA-seq | Unexplored modality combination in breast cancer |
| N4 | Multi-task: subtype + survival + grade | One shared encoder, three simultaneous predictions | Efficient and clinically meaningful — one model, three answers |
| N5 | Per-patient modality attention as explainability | Which data type mattered most for this specific prediction | Directly interpretable for clinicians; not just global feature importance |
| N6 | Uncertainty-aware AI report generation | Flags uncertain cases and generates full structured clinical report via Groq/Llama 3.3 70B | Synthesises all pipeline outputs — subtype, survival, grade, SHAP, attention, drug candidates — into one clinician-readable document |
| N7 | RNA-to-drug precision medicine loop | Differential expression → GDSC drug sensitivity → AlphaFold2 protein structure → DeepPurpose binding affinity | Extends beyond diagnosis and prognosis into actionable treatment recommendation — no other breast cancer fusion paper does this |

---

## 8. Evaluation Metrics

| Task / Aspect | Metric |
|---|---|
| Subtype classification (3-class: HR+/HER2+/TNBC) | Accuracy, Macro F1, AUC (one-vs-rest) |
| Survival prediction | C-index (concordance index) |
| Grade classification (3-class) | Accuracy, Macro F1 |
| Contrastive pretraining alignment | UMAP cluster separation (visual) |
| Uncertainty calibration | Expected Calibration Error (ECE), reliability curve |
| Fairness | Per-subgroup F1 and C-index across age, stage, race, receptor status |
| Drug-target binding (Phase 10) | DeepPurpose predicted binding affinity score |

---

## 9. Folder Structure

```
onco-fusion/
│
├── data/
│   ├── tcga_brca/                           ← Dataset 1 (Kaggle)
│   │   ├── clinical.csv                     ← TCGA patient clinical records
│   │   ├── mrna_expression.csv              ← RNA-seq gene expression
│   │   └── images/                          ← Histopathology patches
│   └── gdsc/                                ← Dataset 2 (Kaggle)
│       └── gdsc_drug_sensitivity.csv        ← IC50 drug response data
│
├── notebooks/
│   ├── 01_EDA.ipynb                         ← Phase 1
│   ├── 02_unimodal_baselines.ipynb          ← Phase 2
│   ├── 03_early_fusion.ipynb                ← Phase 3a ✓
│   ├── 04_late_fusion.ipynb                 ← Phase 3b ✓
│   ├── 05_cross_attention_fusion.ipynb      ← Phase 3c (main model) ✓
│   ├── 06_multitask.ipynb                   ← Phase 4
│   ├── 07_modality_dropout.ipynb            ← Phase 5 (Novelty 1)
│   ├── 08_contrastive_pretrain.ipynb        ← Phase 6 (Novelty 2)
│   ├── 09_explainability.ipynb              ← Phase 7
│   ├── 10_uncertainty_fairness.ipynb        ← Phase 8
│   ├── 11_drug_discovery.ipynb              ← Phase 9 (RNA → Drug targets)
│   └── 12_llm_report.ipynb                 ← Phase 10 (LLM clinical report — final)
│
├── src/
│   ├── encoders/
│   │   ├── image_encoder.py                 ← BiomedCLIP / Phikon / PLIP wrapper
│   │   ├── genomic_encoder.py               ← 1D-CNN encoder
│   │   ├── clinical_encoder.py              ← TabNet encoder
│   │   └── text_encoder.py                  ← BioClinicalBERT wrapper
│   ├── fusion/
│   │   ├── early_fusion.py                  ← Concatenation + MLP
│   │   ├── late_fusion.py                   ← Ensemble of independent classifiers
│   │   └── cross_attention_fusion.py        ← Main model
│   └── tasks/
│       ├── subtype_head.py                  ← 4-class classifier
│       ├── survival_head.py                 ← Cox loss regression
│       └── grade_head.py                    ← 3-class classifier
│
├── app/
│   ├── api.py                               ← FastAPI backend (port 8000)
│   └── frontend.py                          ← Streamlit frontend (port 8501)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                                     ← API keys (gitignored)
├── .gitignore
└── PROJECT_PROPOSAL.md
```

---

## 10. Dependencies

```bash
# Deep learning — models and training
pip install torch torchvision transformers open_clip_torch pytorch-tabnet

# Survival analysis
pip install lifelines

# Classical ML and statistics
pip install scikit-learn xgboost lightgbm scipy

# Explainability
pip install shap

# Dimensionality reduction and visualisation
pip install umap-learn matplotlib seaborn plotly

# Protein structure and drug discovery (Phase 10)
pip install py3Dmol DeepPurpose

# LLM clinical report generation
pip install groq

# Data and Kaggle
pip install python-dotenv kagglehub requests pandas numpy Pillow tqdm

# Deployment
pip install streamlit fastapi uvicorn
```

**Single-line install:**

```bash
pip install torch torchvision transformers open_clip_torch pytorch-tabnet lifelines \
    scikit-learn xgboost lightgbm scipy shap umap-learn matplotlib seaborn plotly \
    py3Dmol groq python-dotenv kagglehub requests pandas numpy Pillow tqdm \
    streamlit fastapi uvicorn
```

---

## Summary

| Aspect | Detail |
|---|---|
| Datasets | 2 (TCGA-BRCA multi-modal fusion + GDSC drug sensitivity) |
| Input modalities | 4 (histopathology image, genomic RNA-seq, clinical tabular, synthetic text) |
| Prediction tasks | 3 simultaneous (subtype, survival, grade) |
| Core model | Cross-attention multi-modal fusion transformer |
| Key novelties | Modality dropout + image-genomics contrastive pretraining + per-patient attention explainability |
| Phase 9 | Drug target identification and repurposing (RNA → GDSC → AlphaFold2 → DeepPurpose) |
| Phase 10 (Final) | LLM clinical report generation via Groq/Llama 3.3 70B — synthesises all pipeline outputs |
| External APIs | HuggingFace (free pretrained models) + Groq (free LLM reports) |

*This project is a complete precision medicine pipeline — from raw multi-modal patient data through subtype classification, survival prediction, explainability, fairness auditing, drug target identification, and treatment recommendation — culminating in an LLM-generated clinical report that synthesises every output into a single document a clinician can act on.*
