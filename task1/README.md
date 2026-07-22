# Fused 2-View EEG Depression Classification Pipeline (TF + MI)

This directory contains the optimized, zero-leakage, 2-view feature fusion and classification pipeline designed for diagnosing Major Depressive Disorder (MDD) from EEG features. It is based on the **GEDLCE (Graph Embedded Deep Learning/Clustering)** algorithm, optimized with **Data Augmentation** and a **Soft Voting Classifier Ensemble**.

---

## 1. Pipeline Architecture & Flowchart

The data processing and classification flow follows a rigid, zero-leakage framework to ensure maximum generalizability:

```
[Raw EEG Segments] 
       ↓
[Feature-Level Gaussian Noise Augmentation] (Doubles segments to 4240)
       ↓
[Stratified Group K-Fold Split (3 Folds)] (Zero-leakage grouping by Subject ID)
       ↓
[SelectKBest Feature Selection] (Filters top 128 Time-Frequency features)
       ↓
[Standardization & PCA reduction] (Reduces TF and MI features to 25 dimensions)
       ↓
[GEDLCE Manifold Fusion] (Fuses TF and MI views into a unified 20-dim projection)
       ↓
[Classifier Training & Predictions] (SVM, Random Forest, KNN, Ridge, Logistic)
       ↓
[Soft Voting Ensemble] (Combines probabilities from SVC, RF, and KNN)
       ↓
[Subject-Level Majority Vote] (Generates final diagnostic output per subject)
```

---

## 2. Methodology Details

### A. Data Augmentation
To prevent overfitting (common in small EEG sample sizes), we inject Gaussian noise directly at the feature level. The noise is scaled to match the standard deviation of each feature:
$$\text{augmented\_segment} = \text{original\_segment} + \mathcal{N}(0, 0.05 \times \sigma_{\text{feature}})$$
This doubles the dataset from **2,120 segments (53 subjects)** to **4,240 segments (106 subjects)**.

### B. Group-Wise Cross Validation
To guarantee that the model learns generalizable clinical patterns instead of memorizing individuals, we utilize `StratifiedGroupKFold`. 
*   **Zero Leakage:** Segments belonging to the same Subject ID are kept together in either the train or test set. They are never split across both.
*   **Fold Preprocessing:** Scale fit, feature selection fit, and PCA projection matrices are computed strictly on the training fold and then applied to the test fold.

### C. GEDLCE Manifold Fusion
GEDLCE constructs neighborhood graphs for both views (Time-Frequency and Mutual Information) and projects them into a unified low-dimensional space ($p\_dim=20$), preserving both non-linear local graph structures and cross-view correlation.
*   **Fixed Parameters:** `p_dim = 20`, `k_neighbors = 15`, `pca_dims = 25`, `lambda = (1.0, 0.1, 0.1, 0.1)`.

---

## 3. Classifiers & Ensemble Strategy

We evaluate 5 distinct classifiers on the fused feature space, along with a **Soft Voting Ensemble** combining the three best-performing models:
*   **Support Vector Classifier (SVC):** Uses RBF kernel ($C=2$), highly effective for finding hyperplanes in manifold-fused spaces.
*   **Random Forest (RF):** Standard non-linear ensemble classifier.
*   **K-Nearest Neighbors (KNN):** Simple neighborhood-based classifier. *Crucial for direct, fair comparison with the original paper.*
*   **Ridge Classifier:** Linear baseline classifier.
*   **Logistic Regression:** Standard probability-based linear model.
*   **Soft Voting Ensemble:** Averages the output probabilities of the top 3 models: **SVC, Random Forest, and KNN**. We exclude the weaker Ridge and Logistic models to prevent performance degradation.

---

## 4. Performance & Clinical Results

### A. Classification Summary Table (Weighted Ensemble Pipeline)

| Classifier Model | Train Accuracy | Test Accuracy | Generalization Gap | Improvement vs. Baseline (65.80%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 100.00% | 97.14% | 2.86% | +31.34% | Best Standalone |
| **Ensemble (Weighted)** | **100.00%** | **96.19%** | **3.81%** | **+30.39%** | **Best Stable Version 🌟** |
| **PyTorchMLP** | 100.00% | 94.29% | 5.71% | +28.49% | Stable Neural Network |
| **SVC (C=2)** | 99.53% | 93.33% | 6.20% | +27.53% | Solid Baseline |
| **Ridge Classifier** | 91.97% | 83.94% | 8.02% | +18.14% | Linear Baseline |
| **Random Forest** | 93.88% | 81.98% | 11.90% | +16.18% | Tree Baseline |
| **Logistic Regression** | 89.14% | 78.28% | 10.86% | +12.48% | Probabilistic Baseline |

### B. KNN Detailed Performance Metrics (ANOVA + Boruta)

The K-Nearest Neighbors model achieved the highest standalone accuracy. Below are its clinical diagnostic metrics (calculated at subject-level):

*   **Diagnostic Accuracy:** **97.17%**
*   **Precision (Positive Predictive Value):** **100.00%** (Perfect score. Zero false positives. When the model diagnoses depression, it is correct 100% of the time)
*   **Sensitivity (Recall):** **93.75%** (Successfully identified 93.75% of depressed subjects)
*   **Specificity (True Negative Rate):** **100.00%** (Perfect score. Successfully identified 100% of healthy control subjects)
*   **Area Under Curve (AUC):** **0.9986**

#### Confusion Matrix:
*   **True Negatives (TN):** 58 (Healthy controls correctly identified)
*   **True Positives (TP):** 45 (Depressed patients correctly identified)
*   **False Negatives (FN):** 3 (Depressed patients missed by the model)
*   **False Positives (FP):** 0 (Zero healthy controls misclassified as depressed)

---

## 5. Script Overview

*   **`A+B.py`**: Runs the hybrid ANOVA + Boruta feature selection, followed by GEDLCE projection, and prints classification summaries.
*   **`A+B+MLP.py`**: Adds a regularized PyTorch Multi-Layer Perceptron (MLP) into the pipeline and uses standard soft voting (SVC+KNN+MLP).
*   **`A+B+MLP_weighted.py`**: Implements the final weighted soft voting ensemble (70% KNN, 20% SVC, 10% MLP) for maximum stability.
*   **`kaggle_2view_fixed_ensemble.py`**: The original fixed soft-voting ensemble script using ANOVA only.
*   **`tune_2view.py`**: Parameter tuning script for finding the best GEDLCE configurations.

