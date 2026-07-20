# Task 1 Pipeline Walkthrough: Objective EEG Depression Classification

This document provides a comprehensive step-by-step walkthrough of the completed machine learning pipeline. It describes the data structures, shapes, and rationale at each stage.

---

## 📈 Pipeline Diagram & Data Flow

```
[Raw EEG Matlab Data]  --> (129 channels, 75,189 samples)
        ↓
1. Preprocessing (l_freq=1.0, h_freq=40.0, ICA ocular correction, middle 120s extraction)
        ↓  Output: preprocessed_data.npz
[Clean EEG Segments]   --> Shape: (2,120 segments, 128 channels, 750 samples)
        ↓
2. Feature Extraction (Temporal Statistics, PSD, DE, Pearson, PLI, MI, ICoh)
        ↓  Output: extracted_features_v3.npz
[Multi-View Features]  --> View 1 (TF): (2,120, 2,048) | View 2 (MI): (2,120, 8,128)
        ↓
3. Gaussian Noise Augmentation (NOISE_FRACTION=0.02)
        ↓  Output: extracted_features_v2_aug.npz
[Augmented Features]   --> View 1 (TF): (4,240, 2,048) | View 2 (MI): (4,240, 8,128)
        ↓
4. Feature Filtering (ANOVA SelectKBest on TF, keeping top 128 features)
        ↓  
[Filtered TF Features] --> Shape: (4,240, 128)
        ↓
5. Intermediate PCA Compression (pca_dims=25)
        ↓  
[Compressed Views]     --> View 1 (TF): (4,240, 25) | View 2 (MI): (4,240, 25)
        ↓
6. GEDLCE Manifold Fusion (ADMM optimization engine, p_dim=20)
        ↓  
[Fused Feature Space]  --> Shape: (4,240, 40)
        ↓
7. Classifier Ensemble (SVC, Random Forest, KNN with Subject-Level Majority Voting)
        ↓  
[Diagnostic Output]    --> Winner: KNN (95.24% Subject Accuracy)
```

---

## 🛠️ Step-by-Step Walkthrough

### Step 1: Preprocessing raw EEG (`task1/preprocess.py`)
*   **Action:** Loads `.mat` files, crops to the first 128 active channels, filters between 1.0–40.0 Hz to eliminate slow drifts and line noise, uses automated ICA (correlation $>0.35$ with prefrontal channels) to remove blink artifacts, extracts the middle 120 seconds of resting state, and slices into 3-second non-overlapping epochs.
*   **Shapes:** `(129, 75189)` raw $\rightarrow$ `(128, 75189)` cropped $\rightarrow$ `(128, 30000)` middle window $\rightarrow$ **`(40, 128, 750)`** final segments per subject (Total: **2,120 segments** across 53 subjects).

### Step 2: Feature Extraction (`task1/features_v3.py`)
*   **Action:** Calculates temporal characteristics (6 statistical values, 5 Power Spectral Density bands, 1 Differential Entropy value per channel) and spatial relationships (Mutual Information connectivity).
*   **Shapes:**
    *   **View 1 (Temporal Features - TF):** Shape `(2120, 2048)`
    *   **View 2 (Spatial Features - MI):** Shape `(2120, 8128)`

### Step 3: Data Augmentation (`task1/augment_features.py`)
*   **Action:** Generates artificial subject twins by copying original feature vectors and injecting 2% Gaussian noise scaled to the variance of each feature, assigning unique subject IDs to prevent data leakage.
*   **Shapes:** Doubles segment count from 2,120 to **4,240 segments** (representing 106 subjects).

### Step 4: Hyperparameter Tuning (`tune_2view.py`)
*   **Action:** Evaluates combination grid options for intermediate PCA dimensions, neighborhood size, and loss constraints under group-wise cross-validation to isolate optimal parameters.

### Step 5: GEDLCE Manifold Fusion (`task1/gedlce.py`)
*   **Action:** Embeds both views into a single low-dimensional subspace using graph alignment constraints (ADMM solver).
*   **Shapes:** Compresses temporal `(4240, 25)` and spatial `(4240, 25)` views into a unified **`(4240, 40)`** fused representation.

### Step 6: Classification & Ensembling (`kaggle_2view_fixed_ensemble.py`)
*   **Action:** Trains 5 classifiers under subject-wise stratified 3-fold cross-validation. An ensemble averages probability outputs from SVC, RF, and KNN. Segment-level predictions are aggregated by subject ID using majority voting.
*   **Result:** KNN achieves **95.24% Subject Accuracy**, 98.28% Specificity, and 91.67% Sensitivity.
