# Subject-Level EEG Depression Detection: Project Overview & Walkthrough

This document contains a comprehensive walkthrough, current status, file directory guide, and validation pipelines for the resting-state EEG Major Depressive Disorder (MDD) detection system.

---

## 📈 Current Project Status Overview
*   **Best Local Performance:** **65.80% ± 10.16% Subject Accuracy** achieved by **Random Forest** under a 3-fold subject-level cross-validation split.
*   **Key Finding:** Skipping ANOVA feature selection causes accuracy to collapse to ~50% (random guess), proving that feature selection is mandatory to counter the curse of dimensionality.
*   **Data Augmentation Status:** Adding 100% artificial Gaussian noise double-data caused the GEDLCE graph fusion iterations to scale non-linearly, running too slowly locally. The current workflow relies on feature selection (ANOVA) and ensembles.

---

## 📂 File Directory Guide (`task1/` & Root)

### 1. Preprocessing Files
*   **`preprocess.py` (Root)**
    *   *What it contains:* Notch filter at 50Hz, bandpass filter at 1.0Hz-40.0Hz, MNE RawArray creation, ICA ocular blink removal, and window segmentation.
    *   *What it does:* Cleans raw MATLAB `.mat` subject data and outputs a segment dataset.
*   **`task1/preprocess_v3.py`**
    *   *What it contains:* Identical preprocessing code.
    *   *What it does:* Configured to point specifically to Kaggle datasets instead of local folders.

### 2. Feature Extraction Files
*   **`task1/features.py`**
    *   *What it contains:* Mean, max, min, variance, skewness, and kurtosis calculations for Temporal Features (TF). Pearson correlation, Phase Lag Index (PLI), and Mutual Information (MI) matrices for Spatial Features (SF).
    *   *What it does:* Generates the baseline feature sets.
*   **`task1/features_v2.py`**
    *   *What it contains:* Computations for extended temporal features (Power Spectral Density, Differential Entropy, Hjorth, Spectral Entropy).
    *   *What it does:* Generates the large 2,688-dimension feature set.
*   **`task1/features_v3.py`**
    *   *What it contains:* Vectorized Imaginary Coherence (ICoh) connectivity math alongside pruned temporal features.
    *   *What it does:* Generates the large 32,512 spatial feature set.

### 3. Graph Fusion & Classifiers
*   **`task1/gedlce.py` / `task1/gedlce_torch.py`**
    *   *What it contains:* Multi-view graph learning algorithm (Graph Eigenvalue Decomposition for Local Connectivity Embedding).
    *   *What it does:* Projects high-dimensional spatial and temporal features into a unified low-dimensional subspace.
*   **`task1/classify_subject_v4.py`**
    *   *What it contains:* ANOVA feature selection (SelectKBest), PCA, GEDLCE projection, and classification ensembles (KNN, RF, XGBoost, SVC, Ridge) with subject-level majority voting.
    *   *What it does:* Executes the main version 4 classification pipeline.

---

## 🛠️ Feature Analysis Configurations

### 1. Baseline Features (`extracted_features.npz` / `extracted_features_v2.npz`)
*   tf = 2,688 dimensions (stats = 768, PSD = 640, DE = 640, Ent = 256, Hjorth = 384)
*   mi = 8,128 dimensions (Mutual Information connectivity)
*   pearson = 8,128 dimensions
*   pli = 8,128 dimensions

### 2. Version 4 Features (After ANOVA Selection)
*   tf = 128 dimensions (Optimized top features selected by ANOVA SelectKBest)
*   mi = 8,128 dimensions
*   pearson = 8,128 dimensions
*   pli = 8,128 dimensions

### 3. Data Augmented Features (Planned Config)
*   tf = 128 dimensions (ANOVA selected from double dataset)
*   mi = 8,128 dimensions
*   Data size = 2x samples (Original + Gaussian Noise generated replicas)

---

## 📌 Pipelines

### Pipeline 1: General File Pipeline
```
[preprocess.py]
      ↓ creates
[preprocessed_data.npz]
      ↓ read by
[features.py]
      ↓ creates
[extracted_features_v2.npz]
      ↓ read by
[classify_subject_v4.py]
      ↓ outputs
[Final Performance Metrics]
```

### Pipeline 2: Subject-Level Cross-Validation Logic
```
53 Subjects
↓
Split subjects first
↓
Train subjects (e.g., 35)
Test subjects (e.g., 18)
↓
Take ALL segments of train subjects
↓
Train on those 3-second segments
↓
Test on ALL segments of unseen subjects
↓
Majority vote → one prediction per subject
```

### Pipeline 3: Version 4 Step-by-Step Architecture
```
[Raw Features: TF=2688, MI=8128]
      ↓
[ANOVA SelectKBest] → Keeps top TF=128
      ↓
[Standard Scaling & PCA] → Reduces to TF=20, MI=20
      ↓
[GEDLCE Graph Fusion] → Fuses views into unified p=10 projection
      ↓
[Classifier Ensembles] → Predictions calculated on test segments
      ↓
[Majority Vote Grouping] → Decides final diagnosis per subject
```
