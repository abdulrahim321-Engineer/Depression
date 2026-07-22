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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
K_TF_COARSE = 500

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else SCRIPT_DIR
v3_aug = os.path.join(OUT_DIR, "extracted_features_v3_aug.npz")

if not os.path.exists(v3_aug):
    print(f"[INFO] Augmented file {v3_aug} not found. Running data augmentation first...")
    import glob
    candidates = [
        os.path.join(WORKING, "extracted_datasets", "extracted_features_v3.npz"),
        os.path.join(SCRIPT_DIR, "extracted_features_v3.npz"),
        os.path.join(SCRIPT_DIR, "..", "extracted_datasets", "extracted_features_v3.npz"),
    ]
    if os.path.exists("/kaggle/input"):
        candidates.extend(glob.glob("/kaggle/input/**/extracted_features_v3.npz", recursive=True))

    v3_src = None
    for c in candidates:
        if os.path.exists(c):
            v3_src = c
            break
    if v3_src is None:
        print(f"[ERROR] Original dataset 'extracted_features_v3.npz' not found in candidates: {candidates}")
        sys.exit(1)
    try:
        from augment_features import augment_file
    except ImportError:
        from task1.augment_features import augment_file
    augment_file(v3_src, v3_aug, copies=1)

feat = np.load(v3_aug)
# Only take the first 768 TF features to be exactly fair with v2 and the original paper
tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
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
            loss = criterion(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()
            
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return np.hstack([1.0 - probs, probs])
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

views_to_test = ["mi", "pli", "pearson", "icoh"]
ensemble_model_names = ["SVC(C=2)", "KNN", "PyTorchMLP"]
uniq = np.unique(subs)
slm = {s: labs[subs == s][0] for s in uniq}
sl = np.array([slm[s] for s in uniq])

results_summary = {}

print("\n" + "=" * 70)
print("  Running Multi-View ANOVA + Boruta + MLP Evaluation")
print("=" * 70)

for view in views_to_test:
    print(f"\n{'='*70}\n  EVALUATING VIEW: TF + {view.upper()}\n{'='*70}")
    
    Xsp_all = np.nan_to_num(feat[view].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    clfs = {
        "SVC(C=2)":       lambda: SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
        "RandomForest":   lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
        "Ridge":          lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "KNN":            lambda: KNeighborsClassifier(n_neighbors=5),
        "LogisticReg":    lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "PyTorchMLP":     lambda: MLPClassifierWrapper(epochs=100, lr=0.005, weight_decay=0.01)
    }
    
    hist_te = {n: [] for n in clfs}
    hist_te["Ensemble"] = []
    
    knn_y_true_all = []
    knn_y_pred_all = []
    knn_y_prob_all = []
    
    ens_y_true_all = []
    ens_y_pred_all = []
    ens_y_prob_all = []

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
        Xtf_tr_r = tf_all[tm]; Xtf_te_r = tf_all[em]
        Xsp_tr   = Xsp_all[tm]; Xsp_te   = Xsp_all[em]
        y_tr = labs[tm]; y_te = labs[em]; tr_s = subs[tm]; te_s = subs[em]

        print(f"\n  [{view.upper()}] Fold {fold+1}:")
        
        # ANOVA
        k_coarse = min(K_TF_COARSE, Xtf_tr_r.shape[1])
        anova = SelectKBest(score_func=f_classif, k=k_coarse)
        Xtf_tr_c = anova.fit_transform(Xtf_tr_r, y_tr)
        Xtf_te_c = anova.transform(Xtf_te_r)

        # Boruta
        try:
            from boruta import BorutaPy
            try:
                from cuml.ensemble import RandomForestClassifier as GPUForest
                rf = GPUForest(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
            except ImportError:
                rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=RANDOM_STATE)
            boruta_sel = BorutaPy(rf, n_estimators='auto', random_state=RANDOM_STATE, verbose=0, max_iter=30)
            boruta_sel.fit(Xtf_tr_c.astype(np.float32), y_tr.astype(np.int32))
            selected_mask = boruta_sel.support_
            if np.sum(selected_mask) == 0: selected_mask = boruta_sel.support_ | boruta_sel.support_weak_
            if np.sum(selected_mask) == 0: selected_mask = (boruta_sel.ranking_ <= 15)
            print(f"    [Boruta] OK (selected {np.sum(selected_mask)} features)")
            Xtf_tr = Xtf_tr_c[:, selected_mask]
            Xtf_te = Xtf_te_c[:, selected_mask]
        except Exception as ex:
            print(f"    [WARNING] Boruta failed ({ex}). Falling back to top 128 ANOVA...")
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

        # GEDLCE
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0, max_iter=30, tol=1e-6,
                   k_neighbors=BEST_PARAMS["k_neighbors"])
        try:
            g.fit([Xtf_tr.T, Xsp_tr.T], y_tr)
        except Exception as ex:
            print(f"    Fold {fold+1} FAILED: {ex}"); continue

        Ft = np.nan_to_num(np.clip(g.transform([Xtf_tr.T, Xsp_tr.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)
        Fe = np.nan_to_num(np.clip(g.transform([Xtf_te.T, Xsp_te.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)

        seg_probs_te = []

        for cn, ci in clfs.items():
            clf = ci()
            clf.fit(Ft, y_tr)
            preds_te = clf.predict(Fe)
            
            if hasattr(clf, "predict_proba"):
                probs_te = clf.predict_proba(Fe)[:, 1]
            else:
                probs_te = clf.decision_function(Fe)
                probs_te = 1.0 / (1.0 + np.exp(-probs_te))

            if cn in ensemble_model_names:
                seg_probs_te.append(probs_te)

            _, sp_te, sq_te = _mv(te_s, preds_te, probs_te)
            _, sy_te = _gt(te_s, y_te)

            te_a = accuracy_score(sy_te, sp_te)
            hist_te[cn].append(te_a)
            print(f"    [{cn:<12}] Test={te_a:.4f}")

            if cn == "KNN":
                knn_y_true_all.extend(sy_te)
                knn_y_pred_all.extend(sp_te)
                knn_y_prob_all.extend(sq_te)

        # Ensemble
        avg_probs_te = np.mean(seg_probs_te, axis=0)
        ens_preds_te = (avg_probs_te >= 0.5).astype(int)

        _, sp_ens_te, sq_ens_te = _mv(te_s, ens_preds_te, avg_probs_te)
        ens_te_a = accuracy_score(sy_te, sp_ens_te)
        hist_te["Ensemble"].append(ens_te_a)
        print(f"    [Ensemble    ] Test={ens_te_a:.4f}")

        ens_y_true_all.extend(sy_te)
        ens_y_pred_all.extend(sp_ens_te)
        ens_y_prob_all.extend(sq_ens_te)
        
    results_summary[view] = {
        "KNN_acc": np.mean(hist_te["KNN"]),
        "Ensemble_acc": np.mean(hist_te["Ensemble"]),
        "SVC_acc": np.mean(hist_te["SVC(C=2)"]),
        "MLP_acc": np.mean(hist_te["PyTorchMLP"])
    }

print(f"\n{'='*70}\n  FINAL SUMMARY ACROSS ALL 4 VIEWS\n{'='*70}")
print(f"{'View':<10} {'KNN Acc':>10} {'Ensemble Acc':>15} {'SVC Acc':>10} {'MLP Acc':>10}")
for view in views_to_test:
    r = results_summary[view]
    print(f"{view.upper():<10} {r['KNN_acc']:>10.4f} {r['Ensemble_acc']:>15.4f} {r['SVC_acc']:>10.4f} {r['MLP_acc']:>10.4f}")

# Plot Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(views_to_test))
w = 0.3
knn_scores = [results_summary[v]["KNN_acc"] for v in views_to_test]
ens_scores = [results_summary[v]["Ensemble_acc"] for v in views_to_test]

ax.bar(x - w/2, knn_scores, w, label="KNN (Best Standalone)", color="#c0504d")
ax.bar(x + w/2, ens_scores, w, label="Weighted Ensemble", color="#4f81bd")

ax.set_title("Cross-View Performance Comparison (EMIRGE Pipeline)")
ax.set_xticks(x)
ax.set_xticklabels([v.upper() for v in views_to_test])
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")

for i, v in enumerate(knn_scores):
    ax.text(i - w/2, v + 0.01, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(ens_scores):
    ax.text(i + w/2, v + 0.01, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot_file = os.path.join(OUT_DIR, "Kaggle_All_Views_Comparison.png")
plt.savefig(plot_file, dpi=150)
print(f"\n[INFO] Comparison Bar Chart saved -> {plot_file}")
print("=" * 70)
