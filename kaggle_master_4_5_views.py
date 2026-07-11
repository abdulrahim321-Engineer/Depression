"""
Kaggle Master Runner — EEG Depression Classification
=====================================================
Run this notebook on Kaggle (GPU T4 x2 recommended).

SETUP BEFORE RUNNING:
  1. Upload extracted_features_v2.npz + extracted_features_v3.npz
     to a Kaggle Dataset called "eeg-depression-features"
  2. Attach that dataset to this notebook
  3. Clone your GitHub repo in Cell 0 below (update the URL)

PIPELINE:
  Step 1: Data Augmentation (2x) — CPU, ~1 min
  Step 2: Tune GEDLCE for 2-view (TF+MI Aug) — GPU, ~15 min
  Step 3: Baseline v4 with tuned params + Aug data — GPU, ~5 min
  Step 4: Tune GEDLCE for 5-view (TF+MI+Pearson+PLI+ICoh Aug) — GPU, ~20 min
  Step 5: Exp 1 (4-view Aug) — GPU, ~10 min
  Step 6: Exp 2 (5-view Aug) — GPU, ~10 min

TOTAL GPU TIME: ~60 minutes (well within 30-hour weekly quota)

All results saved as PNG plots in /kaggle/working/
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 0 — Setup (run this first in a Kaggle Code cell)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Paste this into the FIRST Kaggle code cell:

!git clone https://github.com/YOUR_USERNAME/Depression.git /kaggle/working/Depression
!pip install -q xgboost
import sys
sys.path.insert(0, '/kaggle/working/Depression/task1')
sys.path.insert(0, '/kaggle/working/Depression')
import os
os.chdir('/kaggle/working/Depression')
print("Setup complete!")
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 1 — Copy dataset files from Kaggle input to working directory
# ═══════════════════════════════════════════════════════════════════════════════
import shutil, os

KAGGLE_INPUT = "/kaggle/input/eeg-depression-features"
TASK1_DIR    = "/kaggle/working/Depression/task1"
if os.path.exists(KAGGLE_INPUT):
    os.makedirs(TASK1_DIR, exist_ok=True)
    for f in ["extracted_features_v2.npz", "extracted_features_v3.npz"]:
        src = os.path.join(KAGGLE_INPUT, f)
        dst = os.path.join(TASK1_DIR, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {f}")
        else:
            print(f"Already exists: {f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 2 — Step 1: Data Augmentation (2x)
# ═══════════════════════════════════════════════════════════════════════════════
import os, sys
import numpy as np

# Auto-detect paths
WORKING = "/kaggle/working/Depression" if os.path.exists("/kaggle/working") else os.getcwd()
sys.path.insert(0, os.path.join(WORKING, "task1"))
sys.path.insert(0, WORKING)

print("=" * 60)
print("  STEP 1: Data Augmentation (2x)")
print("=" * 60)

from task1.augment_features import augment_file

TASK1 = os.path.join(WORKING, "task1")
DATASET = "/kaggle/input/datasets/abdulrahim34ew/eeg-depression-features"
if not os.path.exists(DATASET):
    DATASET = TASK1  # Fallback to local task1 folder if not on Kaggle

OUT   = "/kaggle/working" if os.path.exists("/kaggle/working") else TASK1

v2_aug = os.path.join(OUT, "extracted_features_v2_aug.npz")
v3_aug = os.path.join(OUT, "extracted_features_v3_aug.npz")

v2 = os.path.join(DATASET, "extracted_features_v2.npz")
v3 = os.path.join(DATASET, "extracted_features_v3.npz")

# if not os.path.exists(v2_aug):
#     augment_file(v2, v2_aug, copies=1)
# else:
#     print(f"[SKIP] {v2_aug} already exists.")
# 
# if not os.path.exists(v3_aug):
#     augment_file(v3, v3_aug, copies=1)
# else:
#     print(f"[SKIP] {v3_aug} already exists.")

print("[STEP 1 DONE]")


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 3 — Step 2: Tune GEDLCE for 2-view (TF+MI, augmented)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Tune GEDLCE — 2-view (TF+MI Aug)")
print("=" * 60)

def run_tuning_2view(feat_path, out_prefix="tuned_2view"):
    """Run sequential GEDLCE tuning for 2-view (TF+MI) on given feature file."""
    from collections import Counter, defaultdict
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    try:
        from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
        print("[INFO] GPU GEDLCE active")
    except ImportError:
        from gedlce import GEDLCE
        print("[INFO] CPU GEDLCE fallback")

    RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
    N_FOLDS = 3; K_TF = 128; GEDLCE_ITER = 15; BASELINE_ACC = 0.6580

    feat   = np.load(feat_path)
    tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    mi_all = np.nan_to_num(feat["mi"].astype(np.float64),          nan=0.0, posinf=0.0, neginf=0.0)
    labs   = feat["labels"]; subs = feat["subjects"]

    print(f"[INFO] Segments:{tf_all.shape[0]}  Subjects:{np.unique(subs).size}")

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

    best = {"p_dim": 20, "k_neighbors": 10, "pca_dims": 20, "lam": (1.0, 0.1, 0.1, 0.1)}

    for param, space in [
        # ("p_dim",       [5, 10, 15, 20, 25, 30]),
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

    print(f"\n[TUNING 2-VIEW COMPLETE]  Best params: {best}")
    return best


# best_2view = run_tuning_2view(v2_aug, out_prefix="tuned_2view")
# print(f"\nBest 2-view params: {best_2view}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 4 — Step 3: Run Baseline v4 with augmented data + tuned params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3: Baseline v4 (TF+MI, Aug, Tuned)")
print("=" * 60)

# Temporarily point classify_subject_v4.py at the augmented file and tuned params
# We re-use the logic from classify_subject_icoh_baseline.py directly

from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
except ImportError:
    from gedlce import GEDLCE

RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
N_FOLDS = 3; K_TF = 128; BASELINE_ACC = 0.6580

def run_experiment_2view(feat_path, p, spatial_key="mi", exp_name="Exp", plot_path=None):
    feat   = np.load(feat_path, allow_pickle=True)
    tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    sp_all = np.nan_to_num(feat[spatial_key].astype(np.float64),   nan=0.0, posinf=0.0, neginf=0.0)
    labs = feat["labels"]; subs = feat["subjects"]

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
            ss.append(s); ps.append(Counter(pd_[s]).most_common(1)[0][0]); qs.append(float(np.mean(pb_[s])))
        return np.array(ss), np.array(ps), np.array(qs)

    clfs = {
        "SVC(C=2)":     lambda: SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
        "Ridge":        lambda: RidgeClassifier(random_state=RANDOM_STATE),
    }

    hist_tr = {n: [] for n in clfs}; hist_te = {n: [] for n in clfs}
    uniq = np.unique(subs); slm = {s: labs[subs == s][0] for s in uniq}; sl = np.array([slm[s] for s in uniq])
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
        Xtf_tr_r = tf_all[tm]; Xtf_te_r = tf_all[em]
        Xsp_tr   = sp_all[tm]; Xsp_te   = sp_all[em]
        y_tr = labs[tm]; y_te = labs[em]; tr_s = subs[tm]; te_s = subs[em]

        sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_r.shape[1]))
        Xtf_tr = sel.fit_transform(Xtf_tr_r, y_tr); Xtf_te = sel.transform(Xtf_te_r)

        def sps(Xtr, Xte):
            n = min(p["pca_dims"], Xtr.shape[0]-1, Xtr.shape[1])
            s1 = StandardScaler(); pc = PCA(n_components=n, random_state=42); s2 = StandardScaler()
            return (s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))),
                    s2.transform(pc.transform(s1.transform(Xte))), n)

        Xtf_tr, Xtf_te, nc1 = sps(Xtf_tr, Xtf_te)
        Xsp_tr, Xsp_te, nc2 = sps(Xsp_tr, Xsp_te)
        pd_ = min(p["p_dim"], nc1, nc2)
        l0,l1,l2,l3 = p["lam"]
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0, max_iter=30, tol=1e-6,
                   k_neighbors=p["k_neighbors"])
        try:
            g.fit([Xtf_tr.T, Xsp_tr.T], y_tr)
        except Exception as ex:
            print(f"  Fold {fold+1} FAILED: {ex}"); continue

        Ft = np.nan_to_num(np.clip(g.transform([Xtf_tr.T, Xsp_tr.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)
        Fe = np.nan_to_num(np.clip(g.transform([Xtf_te.T, Xsp_te.T]).T, -1e4, 1e4), nan=0.0).astype(np.float32)

        for cn, ci in clfs.items():
            clf = ci(); clf.fit(Ft, y_tr)
            preds = clf.predict(Fe)
            probs = clf.predict_proba(Fe)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(Fe)
            _, sp, _ = _mv(te_s, preds, probs); _, sy = _gt(te_s, y_te)
            te_a = accuracy_score(sy, sp)
            preds_tr = clf.predict(Ft)
            probs_tr = clf.predict_proba(Ft)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(Ft)
            _, spt, _ = _mv(tr_s, preds_tr, probs_tr); _, syt = _gt(tr_s, y_tr)
            tr_a = accuracy_score(syt, spt)
            hist_tr[cn].append(tr_a); hist_te[cn].append(te_a)
            print(f"  Fold{fold+1} [{cn:<12}] Train={tr_a:.4f} Test={te_a:.4f} Gap={tr_a-te_a:+.4f}  vs Baseline={te_a-BASELINE_ACC:+.4f}")

    print(f"\n{'='*70}\n  SUMMARY: {exp_name}\n{'='*70}")
    print(f"{'Classifier':<14} {'Train':>10} {'Test':>10} {'Gap':>8} {'vs Baseline':>14}")
    for cn in clfs:
        tr = np.mean(hist_tr[cn]); te = np.mean(hist_te[cn])
        print(f"{cn:<14} {tr:>10.4f} {te:>10.4f} {tr-te:>+8.4f} {te-BASELINE_ACC:>+12.4f} {'[+]' if te>BASELINE_ACC else '[-]'}")

    if plot_path:
        x = np.arange(len(clfs)); w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x-w/2, [np.mean(hist_tr[n]) for n in clfs], w, label="Train", color="#4f81bd")
        ax.bar(x+w/2, [np.mean(hist_te[n]) for n in clfs], w, label="Test",  color="#c0504d")
        ax.axhline(0.6580, linestyle="--", color="green",  linewidth=2.0, label="Original Baseline RF untuned (65.80%)")
        ax.axhline(0.6057, linestyle="--", color="orange", linewidth=2.0, label="Tuned SVC no-aug (60.57%)")
        ax.set_title(exp_name); ax.set_xticks(x); ax.set_xticklabels(list(clfs.keys()))
        ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(plot_path, dpi=150); print(f"[INFO] Plot saved -> {plot_path}")

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.join(WORKING, "task1")

# run_experiment_2view(
#     v2_aug, p=best_2view, spatial_key="mi",
#     exp_name="Baseline v4 (TF+MI Aug, Tuned)",
#     plot_path=os.path.join(OUT_DIR, "Kaggle_Baseline_v4_Aug_Tuned.png")
# )


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 5 — Step 4: Tune GEDLCE for 5-view (on augmented v3 data)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4: Tune GEDLCE — 5-view (TF+MI+Pearson+PLI+ICoh Aug)")
print("=" * 60)

def run_tuning_nview(feat_path, spatial_keys):
    """Tune GEDLCE for N-view (TF + N spatial features)."""
    try:
        from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
    except ImportError:
        from gedlce import GEDLCE

    RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
    N_FOLDS = 3; K_TF = 128; GEDLCE_ITER = 10

    feat   = np.load(feat_path, allow_pickle=True)
    tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    sp_arrays = [np.nan_to_num(feat[k].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) for k in spatial_keys]
    labs = feat["labels"]; subs = feat["subjects"]
    n_views_total = 1 + len(spatial_keys)

    print(f"[INFO] Segments:{tf_all.shape[0]}  Subjects:{np.unique(subs).size}  Views:{n_views_total}")

    def _gt(sid, ls):
        t = {}
        for s, l in zip(sid, ls):
            if s not in t: t[s] = l
        return np.array(list(t.keys())), np.array(list(t.values()))

    def _mv(sid, preds, probs):
        pd_ = defaultdict(list); pb_ = defaultdict(list)
        for s, pr, q in zip(sid, preds, probs): pd_[s].append(pr); pb_[s].append(q)
        ss, ps, qs = [], [], []
        for s in pd_: ss.append(s); ps.append(Counter(pd_[s]).most_common(1)[0][0]); qs.append(float(np.mean(pb_[s])))
        return np.array(ss), np.array(ps), np.array(qs)

    def run_cv(p_dim, k_neighbors, pca_dims, lam):
        uniq = np.unique(subs); slm = {s: labs[subs == s][0] for s in uniq}; sl = np.array([slm[s] for s in uniq])
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        accs = []
        for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
            tsubs = uniq[tri]; esubs = uniq[tei]
            tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
            Xtf_tr_r = tf_all[tm]; Xtf_te_r = tf_all[em]
            y_tr = labs[tm]; y_te = labs[em]; te_s = subs[em]
            sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_r.shape[1]))
            Xtf_tr = sel.fit_transform(Xtf_tr_r, y_tr); Xtf_te = sel.transform(Xtf_te_r)
            def sps(Xtr, Xte, dims):
                n = min(dims, Xtr.shape[0]-1, Xtr.shape[1])
                s1=StandardScaler(); pc=PCA(n_components=n, random_state=42); s2=StandardScaler()
                return s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))), s2.transform(pc.transform(s1.transform(Xte))), n
            Xtf_tr, Xtf_te, nc = sps(Xtf_tr, Xtf_te, pca_dims)
            views_tr = [Xtf_tr]; views_te = [Xtf_te]; min_nc = nc
            for spa in sp_arrays:
                Xs_tr, Xs_te, ncs = sps(spa[tm], spa[em], pca_dims)
                views_tr.append(Xs_tr); views_te.append(Xs_te); min_nc = min(min_nc, ncs)
            pd_ = min(p_dim, min_nc)
            l0,l1,l2,l3 = lam
            g = GEDLCE(n_views=n_views_total, p_dim=pd_, lambda0=l0, lambda1=l1,
                       lambda2=l2, lambda3=l3, delta=1.0, max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=k_neighbors)
            try:
                g.fit([v.T for v in views_tr], y_tr)
            except Exception:
                accs.append(0.0); continue
            Ft = g.transform([v.T for v in views_tr]).T; Fe = g.transform([v.T for v in views_te]).T
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
            rf.fit(Ft, y_tr)
            preds = rf.predict(Fe); probs = rf.predict_proba(Fe)[:,1]
            _, sp, _ = _mv(te_s, preds, probs); _, sy = _gt(te_s, y_te)
            accs.append(accuracy_score(sy, sp))
        return float(np.mean(accs))

    best = {"p_dim": 20, "k_neighbors": 10, "pca_dims": 20, "lam": (1.0, 0.1, 0.1, 0.1)}
    for param, space in [
        # ("p_dim",       [5, 10, 15, 20, 25, 30]),
        ("k_neighbors", [5, 10, 15, 20]),
        ("pca_dims",    [10, 15, 20, 25]),
        ("lam",         [(1.0,0.1,0.1,0.1),(1.0,0.5,0.5,0.1),(1.0,0.1,0.5,0.5),(2.0,0.1,0.1,0.1)]),
    ]:
        print(f"\n  -- Tuning {param} ({n_views_total}-view) --")
        best_acc, best_val = 0.0, best[param]
        for val in space:
            p = dict(best); p[param] = val
            acc = run_cv(p["p_dim"], p["k_neighbors"], p["pca_dims"], p["lam"])
            print(f"    {param}={str(val):<30}  Acc={acc:.4f}")
            if acc > best_acc: best_acc, best_val = acc, val
        best[param] = best_val
        print(f"  -> Best {param} = {best_val} ({best_acc*100:.2f}%)")

    print(f"\n[TUNING {n_views_total}-VIEW COMPLETE]  Best params: {best}")
    return best


SPATIAL_5VIEW = ["mi", "pearson", "pli", "icoh"]
best_5view = run_tuning_nview(v3_aug, spatial_keys=SPATIAL_5VIEW)
SPATIAL_4VIEW = ["mi", "pearson", "icoh"]
best_4view = run_tuning_nview(v3_aug, spatial_keys=SPATIAL_4VIEW)


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL 6 — Step 5 & 6: Run Exp 1 (4-view) and Exp 2 (5-view) with tuned params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 & 6: Exp 1 (4-view) + Exp 2 (5-view) — Aug + Tuned")
print("=" * 60)

def run_experiment_nview(feat_path, spatial_keys, p, exp_name, plot_path=None):
    try:
        from gedlce_torch import GEDLCETorch as GEDLCE  # type: ignore
    except ImportError:
        from gedlce import GEDLCE

    RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
    N_FOLDS = 3; K_TF = 128; BASELINE_ACC = 0.6580

    feat   = np.load(feat_path, allow_pickle=True)
    tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    sp_arrays = [np.nan_to_num(feat[k].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) for k in spatial_keys]
    labs = feat["labels"]; subs = feat["subjects"]
    n_views = 1 + len(spatial_keys)

    def _gt(sid, ls):
        t = {}
        for s, l in zip(sid, ls):
            if s not in t: t[s] = l
        return np.array(list(t.keys())), np.array(list(t.values()))

    def _mv(sid, preds, probs):
        pd_ = defaultdict(list); pb_ = defaultdict(list)
        for s, pr, q in zip(sid, preds, probs): pd_[s].append(pr); pb_[s].append(q)
        ss, ps, qs = [], [], []
        for s in pd_: ss.append(s); ps.append(Counter(pd_[s]).most_common(1)[0][0]); qs.append(float(np.mean(pb_[s])))
        return np.array(ss), np.array(ps), np.array(qs)

    clfs = {
        "SVC(C=2)":     lambda: SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
        "Ridge":        lambda: RidgeClassifier(random_state=RANDOM_STATE),
    }

    hist_tr = {n: [] for n in clfs}; hist_te = {n: [] for n in clfs}
    uniq = np.unique(subs); slm = {s: labs[subs == s][0] for s in uniq}; sl = np.array([slm[s] for s in uniq])
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)
        Xtf_tr_r = tf_all[tm]; Xtf_te_r = tf_all[em]
        y_tr = labs[tm]; y_te = labs[em]; tr_s = subs[tm]; te_s = subs[em]

        sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_r.shape[1]))
        Xtf_tr = sel.fit_transform(Xtf_tr_r, y_tr); Xtf_te = sel.transform(Xtf_te_r)

        def sps(Xtr, Xte):
            n = min(p["pca_dims"], Xtr.shape[0]-1, Xtr.shape[1])
            s1=StandardScaler(); pc=PCA(n_components=n, random_state=42); s2=StandardScaler()
            return s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))), s2.transform(pc.transform(s1.transform(Xte))), n

        Xtf_tr, Xtf_te, min_nc = sps(Xtf_tr, Xtf_te)
        views_tr = [Xtf_tr]; views_te = [Xtf_te]
        for spa in sp_arrays:
            Xs_tr, Xs_te, ncs = sps(spa[tm], spa[em])
            views_tr.append(Xs_tr); views_te.append(Xs_te); min_nc = min(min_nc, ncs)

        pd_ = min(p["p_dim"], min_nc)
        l0,l1,l2,l3 = p["lam"]
        g = GEDLCE(n_views=n_views, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0, max_iter=30, tol=1e-6,
                   k_neighbors=p["k_neighbors"])
        print(f"  Fold {fold+1}/{N_FOLDS} GEDLCE ({n_views}-view)...", end=" ", flush=True)
        try:
            g.fit([v.T for v in views_tr], y_tr); print("OK")
        except Exception as ex:
            print(f"FAILED: {ex}"); continue

        Ft = np.nan_to_num(np.clip(g.transform([v.T for v in views_tr]).T, -1e4, 1e4), nan=0.0).astype(np.float32)
        Fe = np.nan_to_num(np.clip(g.transform([v.T for v in views_te]).T, -1e4, 1e4), nan=0.0).astype(np.float32)

        for cn, ci in clfs.items():
            clf = ci(); clf.fit(Ft, y_tr)
            preds = clf.predict(Fe)
            probs = clf.predict_proba(Fe)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(Fe)
            _, sp, _ = _mv(te_s, preds, probs); _, sy = _gt(te_s, y_te)
            te_a = accuracy_score(sy, sp)
            preds_tr = clf.predict(Ft)
            probs_tr = clf.predict_proba(Ft)[:,1] if hasattr(clf,"predict_proba") else clf.decision_function(Ft)
            _, spt, _ = _mv(tr_s, preds_tr, probs_tr); _, syt = _gt(tr_s, y_tr)
            tr_a = accuracy_score(syt, spt)
            hist_tr[cn].append(tr_a); hist_te[cn].append(te_a)
            print(f"  [{cn:<12}] Train={tr_a:.4f} Test={te_a:.4f} Gap={tr_a-te_a:+.4f}  vs Baseline={te_a-BASELINE_ACC:+.4f}")

    print(f"\n{'='*70}\n  SUMMARY: {exp_name}\n{'='*70}")
    for cn in clfs:
        tr = np.mean(hist_tr[cn]); te = np.mean(hist_te[cn])
        print(f"{cn:<14} Train={tr:.4f} Test={te:.4f} Gap={tr-te:+.4f} vs Baseline={te-BASELINE_ACC:+.4f} {'[+]' if te>BASELINE_ACC else '[-]'}")

    if plot_path:
        x = np.arange(len(clfs)); w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x-w/2, [np.mean(hist_tr[n]) for n in clfs], w, label="Train", color="#4f81bd")
        ax.bar(x+w/2, [np.mean(hist_te[n]) for n in clfs], w, label="Test",  color="#c0504d")
        ax.axhline(0.6580, linestyle="--", color="green",  linewidth=2.0, label="Original Baseline RF untuned (65.80%)")
        ax.axhline(0.6057, linestyle="--", color="orange", linewidth=2.0, label="Tuned SVC no-aug (60.57%)")
        ax.set_title(exp_name); ax.set_xticks(x); ax.set_xticklabels(list(clfs.keys()))
        ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(plot_path, dpi=150); print(f"[INFO] Plot -> {plot_path}")


run_experiment_nview(
    v3_aug, spatial_keys=["mi","pearson","icoh"], p=best_4view,
    exp_name="Exp 1: 4-view TF+MI+Pearson+ICoh (Aug, Tuned)",
    plot_path=os.path.join(OUT_DIR, "Kaggle_Exp1_4view_Aug_Tuned.png")
)

run_experiment_nview(
    v3_aug, spatial_keys=["mi","pearson","pli","icoh"], p=best_5view,
    exp_name="Exp 2: 5-view TF+MI+Pearson+PLI+ICoh (Aug, Tuned)",
    plot_path=os.path.join(OUT_DIR, "Kaggle_Exp2_5view_Aug_Tuned.png")
)

print("\n" + "=" * 70)
print("  ALL STEPS COMPLETE — Check /kaggle/working/ for PNG plots")
print("=" * 70)
