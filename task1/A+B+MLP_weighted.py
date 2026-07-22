import os
import sys
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Auto-detect paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, WORKING)

try:
    from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
    print("[INFO] GPU GEDLCE active")
except ImportError:
    from gedlce import GEDLCE
    print("[INFO] CPU GEDLCE fallback")

# Setup fixed best parameters from 2-view tuning
BEST_PARAMS = {
    "p_dim": 20,
    "k_neighbors": 15,
    "pca_dims": 25,
    "lam": (1.0, 0.1, 0.1, 0.1)
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
N_FOLDS = 3
K_TF_COARSE = 500  # Number of features selected by ANOVA before Boruta
BASELINE_ACC = 0.6580

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else SCRIPT_DIR
v2_aug = os.path.join(OUT_DIR, "extracted_features_v2_aug.npz")

if not os.path.exists(v2_aug):
    print(f"[INFO] Augmented file {v2_aug} not found. Running data augmentation first...")
    import glob
    candidates = [
        os.path.join(WORKING, "extracted_datasets", "extracted_features_v2.npz"),
        os.path.join(SCRIPT_DIR, "extracted_features_v2.npz"),
        os.path.join(SCRIPT_DIR, "..", "extracted_datasets", "extracted_features_v2.npz"),
    ]
    if os.path.exists("/kaggle/input"):
        candidates.extend(glob.glob("/kaggle/input/**/extracted_features_v2.npz", recursive=True))

    v2_src = None
    for c in candidates:
        if os.path.exists(c):
            v2_src = c
            break
    if v2_src is None:
        print(f"[ERROR] Original dataset 'extracted_features_v2.npz' not found in candidates: {candidates}")
        sys.exit(1)
    try:
        from augment_features import augment_file
    except ImportError:
        from task1.augment_features import augment_file
    augment_file(v2_src, v2_aug, copies=1)

feat = np.load(v2_aug)
tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
mi_all = np.nan_to_num(feat["mi"].astype(np.float64),          nan=0.0, posinf=0.0, neginf=0.0)
labs   = feat["labels"]
subs   = feat["subjects"]

def _gt(sid, ls):
    t = {}
    for s, l in zip(sid, ls):
        if s not in t: t[s] = l
    return np.array(list(t.keys())), np.array(list(t.values()))

def _mv(sid, preds, probs):
    pd_ = defaultdict(list); pb_ = defaultdict(list)
    for s, pr, q in zip(sid, preds, probs):
        pd_[s].append(pr); pb_[s].append(q)
    ss, ps, qs = [], [], []
    for s in pd_:
        ss.append(s)
        ps.append(Counter(pd_[s]).most_common(1)[0][0])
        qs.append(float(np.mean(pb_[s])))
    return np.array(ss), np.array(ps), np.array(qs)

# --- Define PyTorch MLP Model & Sklearn Wrapper ---
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class MLPClassifierWrapper:
    def __init__(self, epochs=100, lr=0.005, weight_decay=0.01):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.model = PyTorchMLP(X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()
            
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(X_t).cpu().numpy()
        return np.hstack([1.0 - probs, probs])
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# Define base models including our new PyTorch MLP Wrapper
clfs = {
    "SVC(C=2)":       lambda: SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
    "RandomForest":   lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    "Ridge":          lambda: RidgeClassifier(random_state=RANDOM_STATE),
    "KNN":            lambda: KNeighborsClassifier(n_neighbors=5),
    "LogisticReg":    lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "PyTorchMLP":     lambda: MLPClassifierWrapper(epochs=100, lr=0.005, weight_decay=0.01)
}

hist_tr = {n: [] for n in clfs}
hist_te = {n: [] for n in clfs}
hist_tr["Ensemble"] = []
hist_te["Ensemble"] = []

uniq = np.unique(subs)
slm = {s: labs[subs == s][0] for s in uniq}
sl = np.array([slm[s] for s in uniq])
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Collect base model and Ensemble predictions across all folds for detailed metrics
knn_y_true_all = []
knn_y_pred_all = []
knn_y_prob_all = []
ens_y_true_all = []
ens_y_prob_all = []

print("\n" + "=" * 60)
print("  Running ANOVA + Boruta + MLP (PyTorch) Weighted Ensemble Pipeline")
print("=" * 60)

for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
    tsubs = uniq[tri]; esubs = uniq[tei]
    tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
    Xtf_tr_r = tf_all[tm]; Xtf_te_r = tf_all[em]
    Xsp_tr   = mi_all[tm]; Xsp_te   = mi_all[em]
    y_tr = labs[tm]; y_te = labs[em]; tr_s = subs[tm]; te_s = subs[em]

    print(f"\nFold {fold+1}:")
    
    # 1. Coarse filter using ANOVA
    k_coarse = min(K_TF_COARSE, Xtf_tr_r.shape[1])
    anova = SelectKBest(score_func=f_classif, k=k_coarse)
    Xtf_tr_c = anova.fit_transform(Xtf_tr_r, y_tr)
    Xtf_te_c = anova.transform(Xtf_te_r)

    # 2. Fine-grained filter using Boruta
    try:
        from boruta import BorutaPy
        try:
            from cuml.ensemble import RandomForestClassifier as GPUForest
            rf = GPUForest(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
            print(f"  [Boruta] Fitting GPU-accelerated Boruta (RAPIDS cuML) on {k_coarse} features...", end=" ", flush=True)
        except ImportError:
            rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE)
            print(f"  [Boruta] Fitting CPU Boruta on {k_coarse} features...", end=" ", flush=True)
        boruta_sel = BorutaPy(rf, n_estimators='auto', random_state=RANDOM_STATE, verbose=0, max_iter=30)
        boruta_sel.fit(Xtf_tr_c.astype(np.float32), y_tr.astype(np.int32))
        
        selected_mask = boruta_sel.support_
        if np.sum(selected_mask) == 0:
            selected_mask = boruta_sel.support_ | boruta_sel.support_weak_
        if np.sum(selected_mask) == 0:
            selected_mask = (boruta_sel.ranking_ <= 15)
            
        print(f"OK (selected {np.sum(selected_mask)} features)")
        Xtf_tr = Xtf_tr_c[:, selected_mask]
        Xtf_te = Xtf_te_c[:, selected_mask]
        
    except ImportError:
        print("  [WARNING] 'boruta' package not found. Falling back to standard top 128 ANOVA features...")
        anova_fb = SelectKBest(score_func=f_classif, k=min(128, Xtf_tr_r.shape[1]))
        Xtf_tr = anova_fb.fit_transform(Xtf_tr_r, y_tr)
        Xtf_te = anova_fb.transform(Xtf_te_r)
    except Exception as ex:
        print(f"  [WARNING] Boruta failed ({ex}). Falling back to top 128 ANOVA features...")
        anova_fb = SelectKBest(score_func=f_classif, k=min(128, Xtf_tr_r.shape[1]))
        Xtf_tr = anova_fb.fit_transform(Xtf_tr_r, y_tr)
        Xtf_te = anova_fb.transform(Xtf_te_r)

    # Scale and PCA
    def sps(Xtr, Xte):
        n = min(BEST_PARAMS["pca_dims"], Xtr.shape[0]-1, Xtr.shape[1])
        s1 = StandardScaler(); pc = PCA(n_components=n, random_state=42); s2 = StandardScaler()
        return (s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))),
                s2.transform(pc.transform(s1.transform(Xte))), n)

    Xtf_tr, Xtf_te, nc1 = sps(Xtf_tr, Xtf_te)
    Xsp_tr, Xsp_te, nc2 = sps(Xsp_tr, Xsp_te)
    pd_ = min(BEST_PARAMS["p_dim"], nc1, nc2)
    l0, l1, l2, l3 = BEST_PARAMS["lam"]

    # Fusing features using GEDLCE
    g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
               lambda2=l2, lambda3=l3, delta=1.0, max_iter=30, tol=1e-6,
               k_neighbors=BEST_PARAMS["k_neighbors"])
    try:
        g.fit([Xtf_tr.T, Xsp_tr.T], y_tr)
    except Exception as ex:
        print(f"  Fold {fold+1} FAILED: {ex}"); continue

    # Project views
    Ft = np.nan_to_num(np.clip(g.transform([Xtf_tr.T, Xsp_tr.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)
    Fe = np.nan_to_num(np.clip(g.transform([Xtf_te.T, Xsp_te.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)

    probs_dict_tr = {}
    probs_dict_te = {}

    # Include KNN, SVC, and PyTorchMLP in the Ensemble
    ensemble_model_names = ["SVC(C=2)", "KNN", "PyTorchMLP"]

    # Train and evaluate individual models
    for cn, ci in clfs.items():
        clf = ci()
        clf.fit(Ft, y_tr)
        
        preds_tr = clf.predict(Ft)
        preds_te = clf.predict(Fe)
        
        if hasattr(clf, "predict_proba"):
            probs_tr = clf.predict_proba(Ft)[:, 1]
            probs_te = clf.predict_proba(Fe)[:, 1]
        else:
            probs_tr = clf.decision_function(Ft)
            probs_te = clf.decision_function(Fe)
            probs_tr = 1.0 / (1.0 + np.exp(-probs_tr))
            probs_te = 1.0 / (1.0 + np.exp(-probs_te))

        if cn in ensemble_model_names:
            probs_dict_tr[cn] = probs_tr
            probs_dict_te[cn] = probs_te

        _, sp_tr, _ = _mv(tr_s, preds_tr, probs_tr)
        _, sp_te, sq_te = _mv(te_s, preds_te, probs_te)
        _, sy_tr = _gt(tr_s, y_tr)
        _, sy_te = _gt(te_s, y_te)

        tr_a = accuracy_score(sy_tr, sp_tr)
        te_a = accuracy_score(sy_te, sp_te)
        
        hist_tr[cn].append(tr_a)
        hist_te[cn].append(te_a)
        print(f"  Fold{fold+1} [{cn:<12}] Train={tr_a:.4f} Test={te_a:.4f} Gap={tr_a-te_a:+.4f}")

        if cn == "KNN":
            knn_y_true_all.extend(sy_te)
            knn_y_pred_all.extend(sp_te)
            knn_y_prob_all.extend(sq_te)

    # Evaluate the Weighted Voting Ensemble (70% KNN, 20% SVC, 10% PyTorchMLP)
    weighted_probs_tr = 0.70 * probs_dict_tr["KNN"] + 0.20 * probs_dict_tr["SVC(C=2)"] + 0.10 * probs_dict_tr["PyTorchMLP"]
    weighted_probs_te = 0.70 * probs_dict_te["KNN"] + 0.20 * probs_dict_te["SVC(C=2)"] + 0.10 * probs_dict_te["PyTorchMLP"]
    
    ens_preds_tr = (weighted_probs_tr >= 0.5).astype(int)
    ens_preds_te = (weighted_probs_te >= 0.5).astype(int)

    _, sp_ens_tr, _ = _mv(tr_s, ens_preds_tr, weighted_probs_tr)
    _, sp_ens_te, sq_ens_te = _mv(te_s, ens_preds_te, weighted_probs_te)
    _, sy_tr = _gt(tr_s, y_tr)
    _, sy_te = _gt(te_s, y_te)

    ens_tr_a = accuracy_score(sy_tr, sp_ens_tr)
    ens_te_a = accuracy_score(sy_te, sp_ens_te)
    hist_tr["Ensemble"].append(ens_tr_a)
    hist_te["Ensemble"].append(ens_te_a)
    print(f"  Fold{fold+1} [Ensemble    ] Train={ens_tr_a:.4f} Test={ens_te_a:.4f} Gap={ens_tr_a-ens_te_a:+.4f}")

    ens_y_true_all.extend(sy_te)
    ens_y_prob_all.extend(sq_ens_te)

# Compute and print KNN detailed metrics
knn_y_true_all = np.array(knn_y_true_all)
knn_y_pred_all = np.array(knn_y_pred_all)
knn_y_prob_all = np.array(knn_y_prob_all)

knn_acc = accuracy_score(knn_y_true_all, knn_y_pred_all)
knn_prec = precision_score(knn_y_true_all, knn_y_pred_all)
knn_sens = recall_score(knn_y_true_all, knn_y_pred_all)
tn, fp, fn, tp = confusion_matrix(knn_y_true_all, knn_y_pred_all).ravel()
knn_spec = tn / (tn + fp)
knn_auc = roc_auc_score(knn_y_true_all, knn_y_prob_all)

print(f"\n{'='*70}\n  KNN DETAILED PERFORMANCE METRICS (Subject-Level)\n{'='*70}")
print(f"  Accuracy:    {knn_acc:.4f}")
print(f"  Precision:   {knn_prec:.4f}")
print(f"  Sensitivity: {knn_sens:.4f} (Recall)")
print(f"  Specificity: {knn_spec:.4f}")
print(f"  AUC Score:   {knn_auc:.4f}")
print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Final Summary Table
print(f"\n{'='*70}\n  SUMMARY: ANOVA + Boruta + MLP (PyTorch) Weighted Ensemble Experiment\n{'='*70}")
print(f"{'Classifier':<15} {'Train':>10} {'Test':>10} {'Gap':>8} {'vs Baseline':>14}")
all_names = list(clfs.keys()) + ["Ensemble"]
for cn in all_names:
    tr = np.mean(hist_tr[cn])
    te = np.mean(hist_te[cn])
    print(f"{cn:<15} {tr:>10.4f} {te:>10.4f} {tr-te:>+8.4f} {te-BASELINE_ACC:>+12.4f} {'[+]' if te>BASELINE_ACC else '[-]'}")

# Plotting the results
x = np.arange(len(all_names))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x-w/2, [np.mean(hist_tr[n]) for n in all_names], w, label="Train", color="#4f81bd")
ax.bar(x+w/2, [np.mean(hist_te[n]) for n in all_names], w, label="Test",  color="#c0504d")
ax.axhline(BASELINE_ACC, linestyle="--", color="green", linewidth=2.0, label=f"Original Baseline RF ({BASELINE_ACC*100:.1f}%)")
ax.set_title("ANOVA + Boruta + MLP 2-View Classifier & Weighted Ensemble Performance")
ax.set_xticks(x)
ax.set_xticklabels(all_names)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
plt.tight_layout()
plot_file = os.path.join(OUT_DIR, "Kaggle_2view_Boruta_MLP_Weighted_Ensemble.png")
plt.savefig(plot_file, dpi=150)
print(f"\n[INFO] Plot saved -> {plot_file}")

# Save Confusion Matrix Heatmap for KNN
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(knn_y_true_all, knn_y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Depressed"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("KNN Confusion Matrix (Subject-Level)")
cm_plot_file = os.path.join(OUT_DIR, "Kaggle_2view_Boruta_MLP_Weighted_KNN_Confusion_Matrix.png")
plt.tight_layout()
plt.savefig(cm_plot_file, dpi=150)
print(f"[INFO] Confusion Matrix Plot saved -> {cm_plot_file}")

# Save ROC Curves for KNN and Ensemble
fig, ax = plt.subplots(figsize=(7, 6))

fpr_knn, tpr_knn, _ = roc_curve(knn_y_true_all, knn_y_prob_all)
roc_auc_knn = auc(fpr_knn, tpr_knn)
ax.plot(fpr_knn, tpr_knn, color="darkorange", lw=2, label=f"KNN ROC (AUC = {roc_auc_knn:.4f})")

fpr_ens, tpr_ens, _ = roc_curve(np.array(ens_y_true_all), np.array(ens_y_prob_all))
roc_auc_ens = auc(fpr_ens, tpr_ens)
ax.plot(fpr_ens, tpr_ens, color="navy", lw=2, label=f"Ensemble ROC (AUC = {roc_auc_ens:.4f})")

ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curves")
ax.legend(loc="lower right")
roc_plot_file = os.path.join(OUT_DIR, "Kaggle_2view_Boruta_MLP_Weighted_ROC_Curves.png")
plt.tight_layout()
plt.savefig(roc_plot_file, dpi=150)
print(f"[INFO] ROC Curves Plot saved -> {roc_plot_file}")
print("=" * 70)
