import os
import sys
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Auto-detect paths
WORKING = "/kaggle/working/Depression" if os.path.exists("/kaggle/working") else os.getcwd()
sys.path.insert(0, os.path.join(WORKING, "task1"))
sys.path.insert(0, WORKING)

try:
    from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
    print("[INFO] GPU GEDLCE active")
except ImportError:
    from gedlce import GEDLCE
    print("[INFO] CPU GEDLCE fallback")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_FOLDS = 3
K_TF = 128
GEDLCE_ITER = 15  # Optimized to 15 iterations for fast tuning
BASELINE_ACC = 0.6580

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.join(WORKING, "task1")
v2_aug = os.path.join(OUT_DIR, "extracted_features_v2_aug.npz")

if not os.path.exists(v2_aug):
    print(f"[ERROR] {v2_aug} not found! Please run data augmentation first.")
    sys.exit(1)

feat = np.load(v2_aug)
tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
mi_all = np.nan_to_num(feat["mi"].astype(np.float64),          nan=0.0, posinf=0.0, neginf=0.0)
labs   = feat["labels"]
subs   = feat["subjects"]

print(f"[INFO] Tuning Segments: {tf_all.shape[0]} | Subjects: {np.unique(subs).size}")

def subject_ground_truth(sid, ls):
    t = {}
    for s, l in zip(sid, ls):
        if s not in t: t[s] = l
    return np.array(list(t.keys())), np.array(list(t.values()))

def majority_vote(sid, preds, probs):
    pd_ = defaultdict(list); pb_ = defaultdict(list)
    for s, p, q in zip(sid, preds, probs):
        pd_[s].append(p); pb_[s].append(q)
    ss, ps, qs = [], [], []
    for s in pd_:
        ss.append(s)
        ps.append(Counter(pd_[s]).most_common(1)[0][0])
        qs.append(float(np.mean(pb_[s])))
    return np.array(ss), np.array(ps), np.array(qs)

def run_cv(p_dim, k_neighbors, pca_dims, lam):
    uniq = np.unique(subs)
    slm  = {s: labs[subs == s][0] for s in uniq}
    sl   = np.array([slm[s] for s in uniq])
    cv   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    
    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
        Xtf_tr_raw = tf_all[tm]; Xtf_te_raw = tf_all[em]
        Xmi_tr = mi_all[tm];    Xmi_te = mi_all[em]
        y_train = labs[tm];     y_test  = labs[em]
        test_subs = subs[em]
        
        sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_raw.shape[1]))
        Xtf_tr = sel.fit_transform(Xtf_tr_raw, y_train)
        Xtf_te = sel.transform(Xtf_te_raw)
        
        def sps(Xtr, Xte):
            n = min(pca_dims, Xtr.shape[0]-1, Xtr.shape[1])
            s1 = StandardScaler(); pc = PCA(n_components=n, random_state=42); s2 = StandardScaler()
            return (s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))),
                    s2.transform(pc.transform(s1.transform(Xte))), n)
                    
        Xtf_tr, Xtf_te, nc1 = sps(Xtf_tr, Xtf_te)
        Xmi_tr, Xmi_te, nc2 = sps(Xmi_tr, Xmi_te)
        pd_ = min(p_dim, nc1, nc2)
        l0, l1, l2, l3 = lam
        
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0,
                   max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=k_neighbors)
        try:
            g.fit([Xtf_tr.T, Xmi_tr.T], y_train)
        except Exception:
            accs.append(0.0); continue
            
        Ft = g.transform([Xtf_tr.T, Xmi_tr.T]).T
        Fe = g.transform([Xtf_te.T, Xmi_te.T]).T
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        rf.fit(Ft, y_train)
        preds = rf.predict(Fe)
        probs = rf.predict_proba(Fe)[:, 1]
        
        _, sp, _ = majority_vote(test_subs, preds, probs)
        _, sy    = subject_ground_truth(test_subs, y_test)
        accs.append(accuracy_score(sy, sp))
        
    return float(np.mean(accs))

# Starting default params
best = {"p_dim": 20, "k_neighbors": 10, "pca_dims": 20, "lam": (1.0, 0.1, 0.1, 0.1)}

print("\n" + "=" * 60)
print("  STEP 2: Tuning GEDLCE Parameters (TF+MI)")
print("=" * 60)

for param, space in [
    # ("p_dim",       [5, 10, 15, 20, 25, 30]),  # Fixed to 20
    ("k_neighbors", [5, 10, 15, 20]),
    ("pca_dims",    [10, 15, 20, 25]),
    ("lam",         [(1.0,0.1,0.1,0.1),(1.0,0.5,0.5,0.1),(1.0,0.1,0.5,0.5),(2.0,0.1,0.1,0.1)]),
]:
    print(f"\n  -- Tuning {param} --")
    best_acc, best_val = 0.0, best[param]
    for val in space:
        p = dict(best); p[param] = val
        acc = run_cv(p["p_dim"], p["k_neighbors"], p["pca_dims"], p["lam"])
        sign = "[+]" if acc > BASELINE_ACC else "[-]"
        print(f"    {param}={str(val):<30}  Acc={acc:.4f}  {sign}")
        if acc > best_acc:
            best_acc, best_val = acc, val
    best[param] = best_val
    print(f"  -> Best {param} = {best_val} ({best_acc*100:.2f}%)")

print(f"\n[TUNING COMPLETE] Final Best Parameters: {best}")
