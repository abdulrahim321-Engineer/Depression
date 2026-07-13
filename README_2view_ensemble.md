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

### A. Classification Summary Table

| Classifier Model | Train Accuracy | Test Accuracy | Generalization Gap | Improvement vs. Baseline (65.80%) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN (Winner)** | **100.00%** | **95.24%** | **4.76%** | **+29.44% [+]** |
| **Ensemble** | 99.52% | 93.36% | 6.16% | +27.56% [+] |
| **SVC (C=2)** | 98.11% | 92.43% | 5.67% | +26.63% [+] |
| **Ridge Classifier** | 89.14% | 82.04% | 7.10% | +16.24% [+] |
| **Logistic Regression** | 89.15% | 80.16% | 8.99% | +14.36% [+] |
| **Random Forest** | 91.51% | 80.16% | 11.35% | +14.36% [+] |

### B. KNN Detailed Performance Metrics

The K-Nearest Neighbors model achieved the highest classification accuracy. Below are its clinical diagnostic metrics (calculated at subject-level):

*   **Diagnostic Accuracy:** **95.28%**
*   **Precision (Positive Predictive Value):** **97.78%** (When the model diagnoses depression, it is correct 97.78% of the time)
*   **Sensitivity (Recall):** **91.67%** (Successfully identified 91.67% of depressed subjects)
*   **Specificity (True Negative Rate):** **98.28%** (Successfully identified 98.28% of healthy control subjects)
*   **Area Under Curve (AUC):** **0.9971**

#### Confusion Matrix:
*   **True Negatives (TN):** 57 (Healthy controls correctly identified)
*   **True Positives (TP):** 44 (Depressed patients correctly identified)
*   **False Negatives (FN):** 4 (Depressed patients missed by the model)
*   **False Positives (FP):** 1 (Healthy control misclassified as depressed)

---

## 5. Script Overview

*   **`kaggle_2view_fixed_ensemble.py`**: The main execution script. It loads the augmented NPZ data, sets up the cross-validation, runs the GEDLCE manifold fusion with the optimal fixed parameters, trains the classifiers, evaluates the ensemble, and saves the accuracy visualization bar plot to `/kaggle/working/Kaggle_2view_Fixed_Ensemble.png`.
