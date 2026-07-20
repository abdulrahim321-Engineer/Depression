# Task 1: Complete EEG Depression Classification Guide

This guide provides an in-depth explanation of the mathematical and algorithmic concepts behind our pipeline, focusing on **ANOVA Feature Selection**, **GEDLCE Manifold Fusion**, and the **ADMM Optimization Solver**.

---

## 🔬 1. ANOVA: The Supervised Feature Filter

### What it is & What it does
ANOVA (Analysis of Variance) is a supervised univariate statistical feature selection method. In our pipeline, it evaluates each of the **2,048 temporal features** individually to measure how significantly their values differ between the Healthy Control (HC) and Major Depressive Disorder (MDD) classes.

### The Mathematics (In a Nutshell)
For each feature, ANOVA calculates an **F-value**:
$$F = \frac{\text{Variance between classes (HC vs MDD)}}{\text{Variance within classes}}$$

*   **High F-value:** Means the feature's values differ significantly between healthy and depressed subjects, while staying consistent within each group.
*   **Low F-value:** Means the feature values overlap heavily between groups, indicating it is clinical noise.
*   **The Action:** We sort all 2,048 features by their F-value, drop the bottom 1,920 features, and retain only the **top 128 features**. This drastically reduces computational complexity and removes noisy/irrelevant features before applying PCA.

---

## 🧬 2. GEDLCE: Graph-Enhanced Dual Low-Rank Correlation Embedding

### Is it one equation or multiple?
**GEDLCE is a unified optimization framework defined by a single objective function (Equation 10 in the paper).** 
However, because this single objective function is too complex to solve directly in one step, it is broken down into a set of **multiple coordinate update equations** (solved using the ADMM algorithm). 

The original paper authors (Zhang et al., Lanzhou University, 2026) designed this specific mathematical formulation to fuse spatio-temporal features.

### Breakdown of the Core Name Components

#### **A. Graph-Enhanced (Locality Preservation)**
*   **How it works:** It constructs two local neighborhood graphs: a *Within-Class Graph* (connecting patients with the same diagnosis) and a *Between-Class Graph* (connecting patients with different diagnoses).
*   **Why it's effective:** It forces the fused features of depressed patients to cluster tightly together in the subspace, while pushing healthy controls away, creating a clean boundary that simple classifiers (like KNN) can easily separate.

#### **B. Dual Low-Rank (Noise Reduction)**
*   **How it works:** It applies low-rank constraints to both the feature projection matrix ($P$) and the sample representation matrix ($Z$) using Singular Value Thresholding.
*   **Why it's effective:** Raw EEG contains overlapping, redundant features and background noise. Forcing a low-rank representation compresses the data down to its true, latent neural components, effectively filtering out clinical noise.

#### **C. Correlation Embedding (View Alignment)**
*   **How it works:** It integrates a Canonical Correlation Analysis (CCA) term that maximizes the mathematical correlation between the temporal (TF) view and the spatial (MI) view.
*   **Why it's effective:** It ensures that local brainwave speeds and global communication networks are aligned, preventing them from contradicting each other and combining their diagnostic power.

---

## ⚙️ 3. The ADMM Solver: The Mathematical Engine

### What it does
**ADMM (Alternating Direction Method of Multipliers)** is the optimization solver used to find the projection matrices ($P_1, P_2$) that satisfy the GEDLCE constraints. 

It takes the large, complex GEDLCE objective function and splits it into five smaller, independent sub-problems (solved in an alternating loop):
1.  **Update $J$:** Implements Singular Value Thresholding (SVT) to enforce the low-rank constraint.
2.  **Update $Q$:** Standardizes the projection matrix.
3.  **Update $E$:** Applies a mathematical "shrinkage" operator to isolate sparse noise.
4.  **Update $Z$:** Solves a linear system to update the sample representations.
5.  **Update $P_i$:** Solves a Lyapunov equation for each view to update the projection matrices.

### Why it's effective
Without ADMM, solving the multi-objective GEDLCE equations would be computationally impossible. ADMM guarantees convergence, handles non-differentiable mathematical terms (like sparsity and low-rank matrices), and allows us to run optimizations in parallel on GPU cores (via our PyTorch implementation).
