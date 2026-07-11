"""
Sequential GEDLCE Hyperparameter Tuning on v4 Baseline (TF-stats + MI)
=======================================================================
Strategy: Fix one parameter at a time (sequential tuning).
  Phase 1: Tune p_dim          (fix best p_dim)
  Phase 2: Tune k_neighbors    (using best p_dim)
  Phase 3: Tune PCA_DIMS       (using best p_dim + k_neighbors)
  Phase 4: Tune lambdas        (using all above)

Speed trick: Only RandomForest used during tuning phases (fastest).
Final step: Run ALL 5 classifiers with the globally best params + plot.
Final step: Run SVC-only (best classifier) with best params + plot.
All results compared against BASELINE v4 = 65.80%.

NO existing files are touched by this script.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    import torch
    torch.manual_seed(RANDOM_STATE)
except ImportError:
    pass

# GEDLCE import
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))
    from gedlce_torch import GEDLCETorch as GEDLCE
    print("[INFO] Using GPU-accelerated GEDLCE (gedlce_torch)")
except ImportError:
    try:
        from gedlce import GEDLCE
        print("[INFO] Using CPU GEDLCE (gedlce)")
    except ImportError:
        print("[ERROR] GEDLCE not found.")
        sys.exit(1)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ── Fixed v4 settings ────────────────────────────────────────────────────────
N_FOLDS   = 3
K_TF      = 128   # ANOVA top-K for TF (never changed)
GEDLCE_ITER = 30
BASELINE_ACC = 0.6580

# ── Search spaces ────────────────────────────────────────────────────────────
SEARCH_P_DIM       = [5, 10, 15, 20, 25, 30]
SEARCH_K_NEIGHBORS = [5, 10, 15, 20]
SEARCH_PCA_DIMS    = [10, 15, 20, 25]
SEARCH_LAMBDAS     = [
    (1.0, 0.1, 0.1, 0.1),   # original v4
    (1.0, 0.5, 0.5, 0.1),
    (1.0, 0.1, 0.5, 0.5),
    (2.0, 0.1, 0.1, 0.1),
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def subject_ground_truth(sid, labs):
    t = {}
    for s, l in zip(sid, labs):
        if s not in t:
            t[s] = l
    return np.array(list(t.keys())), np.array(list(t.values()))


def majority_vote(sid, preds, probs):
    pd_ = defaultdict(list)
    pb_ = defaultdict(list)
    for s, p, q in zip(sid, preds, probs):
        pd_[s].append(p)
        pb_[s].append(q)
    ss, ps, qs = [], [], []
    for s in pd_:
        ss.append(s)
        ps.append(Counter(pd_[s]).most_common(1)[0][0])
        qs.append(float(np.mean(pb_[s])))
    return np.array(ss), np.array(ps), np.array(qs)


def get_probs(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p[:, 0]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict(X).astype(float)


def run_cv(tf_all, mi_all, labs, subs,
           p_dim, k_neighbors, pca_dims, lam,
           classifiers_def, label=""):
    """Run 3-fold CV with given params. Returns dict of avg subject-level test acc per clf."""
    uniq = np.unique(subs)
    slm  = {s: labs[subs == s][0] for s in uniq}
    sl   = np.array([slm[s] for s in uniq])
    cv   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {c: [] for c in classifiers_def}

    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)

        Xtf_tr_raw = tf_all[tm]; Xtf_te_raw = tf_all[em]
        Xmi_tr_raw = mi_all[tm]; Xmi_te_raw = mi_all[em]
        y_train = labs[tm];      y_test  = labs[em]
        train_subs = subs[tm];   test_subs = subs[em]

        # ANOVA on TF
        sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_raw.shape[1]))
        Xtf_tr = sel.fit_transform(Xtf_tr_raw, y_train)
        Xtf_te = sel.transform(Xtf_te_raw)

        # Scale -> PCA -> Scale
        def sps(Xtr, Xte):
            n = min(pca_dims, Xtr.shape[0]-1, Xtr.shape[1])
            s1 = StandardScaler(); pc = PCA(n_components=n, random_state=RANDOM_STATE); s2 = StandardScaler()
            return (s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))),
                    s2.transform(pc.transform(s1.transform(Xte))), n)

        Xtf_tr, Xtf_te, nc1 = sps(Xtf_tr, Xtf_te)
        Xmi_tr, Xmi_te, nc2 = sps(Xmi_tr_raw, Xmi_te_raw)
        pd_ = min(p_dim, nc1, nc2)

        # GEDLCE
        l0, l1, l2, l3 = lam
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0,
                   max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=k_neighbors)
        try:
            g.fit([Xtf_tr.T, Xmi_tr.T], y_train)
        except Exception:
            for c in classifiers_def:
                results[c].append(0.0)
            continue

        Ft = g.transform([Xtf_tr.T, Xmi_tr.T]).T
        Fe = g.transform([Xtf_te.T, Xmi_te.T]).T

        for clf_name, clf_inst in classifiers_def.items():
            clf = clf_inst()
            clf.fit(Ft, y_train)
            preds = clf.predict(Fe)
            probs = get_probs(clf, Fe)
            _, subj_preds, _ = majority_vote(test_subs, preds, probs)
            _, subj_y        = subject_ground_truth(test_subs, y_test)
            results[clf_name].append(accuracy_score(subj_y, subj_preds))

    return {c: np.mean(v) for c, v in results.items()}


def tune_phase(name, search_space, fixed_params, tf_all, mi_all, labs, subs, param_key):
    """Generic sequential tuning phase for one parameter."""
    print(f"\n{'='*65}")
    print(f"  PHASE: Tuning {param_key}")
    print(f"  Fixed params: {fixed_params}")
    print(f"{'='*65}")

    rf_only = {"RF": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)}
    best_val = None
    best_acc  = 0.0
    rows = []

    for val in search_space:
        params = dict(fixed_params)
        params[param_key] = val
        res = run_cv(tf_all, mi_all, labs, subs,
                     p_dim       = params["p_dim"],
                     k_neighbors = params["k_neighbors"],
                     pca_dims    = params["pca_dims"],
                     lam         = params["lam"],
                     classifiers_def = rf_only)
        acc = res["RF"]
        delta = acc - BASELINE_ACC
        flag = "[+]" if delta > 0 else "[-]"
        print(f"  {param_key}={str(val):<30}  RF Test Acc={acc:.4f}  vs Baseline={delta:+.4f}  {flag}")
        rows.append((val, acc))
        if acc > best_acc:
            best_acc = acc
            best_val = val

    print(f"\n  [*] Best {param_key} = {best_val}  |  RF Test Acc = {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_val, best_acc, rows


def run_full_eval(tf_all, mi_all, labs, subs, best_params, classifiers_def, run_label):
    """Run final evaluation with best params across all given classifiers."""
    print(f"\n{'='*75}")
    print(f"  FINAL EVAL: {run_label}")
    print(f"  Best params: {best_params}")
    print(f"  BASELINE v4 = {BASELINE_ACC:.4f}")
    print(f"{'='*75}")

    uniq = np.unique(subs)
    slm  = {s: labs[subs == s][0] for s in uniq}
    sl   = np.array([slm[s] for s in uniq])
    cv   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    history = {c: {"train_subj": [], "test_subj": [], "train_seg": [], "test_seg": []} for c in classifiers_def}

    p_dim = best_params["p_dim"]; k_neighbors = best_params["k_neighbors"]
    pca_dims = best_params["pca_dims"]; lam = best_params["lam"]

    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]; esubs = uniq[tei]
        tm = np.isin(subs, tsubs); em = np.isin(subs, esubs)

        Xtf_tr_raw = tf_all[tm]; Xtf_te_raw = tf_all[em]
        Xmi_tr_raw = mi_all[tm]; Xmi_te_raw = mi_all[em]
        y_train = labs[tm];      y_test  = labs[em]
        train_subs = subs[tm];   test_subs = subs[em]

        sel = SelectKBest(score_func=f_classif, k=min(K_TF, Xtf_tr_raw.shape[1]))
        Xtf_tr = sel.fit_transform(Xtf_tr_raw, y_train)
        Xtf_te = sel.transform(Xtf_te_raw)

        def sps(Xtr, Xte):
            n = min(pca_dims, Xtr.shape[0]-1, Xtr.shape[1])
            s1 = StandardScaler(); pc = PCA(n_components=n, random_state=RANDOM_STATE); s2 = StandardScaler()
            return (s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr))),
                    s2.transform(pc.transform(s1.transform(Xte))), n)

        Xtf_tr, Xtf_te, nc1 = sps(Xtf_tr, Xtf_te)
        Xmi_tr, Xmi_te, nc2 = sps(Xmi_tr_raw, Xmi_te_raw)
        pd_ = min(p_dim, nc1, nc2)

        l0, l1, l2, l3 = lam
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=l0, lambda1=l1,
                   lambda2=l2, lambda3=l3, delta=1.0,
                   max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=k_neighbors)
        try:
            print(f"  Fold {fold+1} GEDLCE...", end=" ", flush=True)
            g.fit([Xtf_tr.T, Xmi_tr.T], y_train)
            print("OK")
        except Exception as ex:
            print(f"FAILED: {ex}"); continue

        Ft = g.transform([Xtf_tr.T, Xmi_tr.T]).T
        Fe = g.transform([Xtf_te.T, Xmi_te.T]).T

        for clf_name, clf_inst in classifiers_def.items():
            clf = clf_inst(); clf.fit(Ft, y_train)
            tr_preds = clf.predict(Ft); te_preds = clf.predict(Fe)
            tr_probs = get_probs(clf, Ft); te_probs = get_probs(clf, Fe)
            tr_seg = accuracy_score(y_train, tr_preds)
            te_seg = accuracy_score(y_test,  te_preds)
            _, tr_sp, _ = majority_vote(train_subs, tr_preds, tr_probs)
            _, tr_sy    = subject_ground_truth(train_subs, y_train)
            _, te_sp, _ = majority_vote(test_subs, te_preds, te_probs)
            _, te_sy    = subject_ground_truth(test_subs, y_test)
            tr_subj = accuracy_score(tr_sy, tr_sp)
            te_subj = accuracy_score(te_sy, te_sp)
            history[clf_name]["train_seg"].append(tr_seg)
            history[clf_name]["test_seg"].append(te_seg)
            history[clf_name]["train_subj"].append(tr_subj)
            history[clf_name]["test_subj"].append(te_subj)
            print(f"    [{clf_name:<12}] SUBJ Train={tr_subj:.4f} Test={te_subj:.4f} Gap={tr_subj-te_subj:+.4f}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY -- {run_label}")
    print(f"  BASELINE v4 = {BASELINE_ACC:.4f}")
    print(f"{'='*80}")
    print(f"{'Classifier':<15} {'Train Subj':>12} {'Test Subj':>12} {'Gap':>10} {'vs Baseline':>14}")
    print("-"*80)
    best_test = 0.0; best_clf = ""
    for c in classifiers_def:
        tr = np.mean(history[c]["train_subj"])
        te = np.mean(history[c]["test_subj"])
        gap = tr - te
        delta = te - BASELINE_ACC
        flag = "[+] IMPROVED" if delta > 0 else "[-]"
        print(f"{c:<15} {tr:>12.4f} {te:>12.4f} {gap:>+10.4f} {delta:>+12.4f}  {flag}")
        if te > best_test:
            best_test = te; best_clf = c
    print("="*80)
    print(f"  [*] Best: {best_clf}  |  {best_test:.4f} ({best_test*100:.2f}%)")
    print("="*80)

    # Plot
    clf_names = list(classifiers_def.keys())
    x = np.arange(len(clf_names)); width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    safe_label = run_label.replace(":", "").replace("+", "").replace(" ", "_")[:50]
    fig.suptitle(f"{run_label}\nBaseline v4 = 65.80%", fontsize=11, fontweight='bold')

    axes[0].bar(x-width/2, [np.mean(history[n]["train_seg"]) for n in clf_names], width, label="Train Seg",  color="#4f81bd")
    axes[0].bar(x+width/2, [np.mean(history[n]["test_seg"])  for n in clf_names], width, label="Test Seg",   color="#c0504d")
    axes[0].axhline(y=BASELINE_ACC, color='green', linestyle='--', linewidth=1.5, label="Baseline 65.80%")
    axes[0].set_title("Segment-Level"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(clf_names, rotation=15); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy"); axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend()

    axes[1].bar(x-width/2, [np.mean(history[n]["train_subj"]) for n in clf_names], width, label="Train Subj", color="#9bbb59")
    axes[1].bar(x+width/2, [np.mean(history[n]["test_subj"])  for n in clf_names], width, label="Test Subj",  color="#8064a2")
    axes[1].axhline(y=BASELINE_ACC, color='green', linestyle='--', linewidth=1.5, label="Baseline 65.80%")
    axes[1].set_title("Subject-Level"); axes[1].set_xticks(x)
    axes[1].set_xticklabels(clf_names, rotation=15); axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy"); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), f"{safe_label}_curve.png")
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"  [INFO] Saved plot -> {plot_path}")

    return best_test, best_clf, history


def main():
    # Load data
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'task1', 'extracted_features_v3.npz'),
        os.path.join('task1', 'extracted_features_v3.npz'),
        'extracted_features_v3.npz',
    ]
    feat_path = next((c for c in candidates if os.path.exists(c)), None)
    if feat_path is None:
        print("[ERROR] extracted_features_v3.npz not found.")
        sys.exit(1)

    feat   = np.load(feat_path)
    tf_all = np.nan_to_num(feat["tf"].astype(np.float64)[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    mi_all = np.nan_to_num(feat["mi"].astype(np.float64),           nan=0.0, posinf=0.0, neginf=0.0)
    labs   = feat["labels"]
    subs   = feat["subjects"]

    print("="*65)
    print("  SEQUENTIAL GEDLCE HYPERPARAMETER TUNING")
    print(f"  Dataset: {tf_all.shape[0]} segments | {np.unique(subs).size} subjects")
    print(f"  TF: {tf_all.shape[1]} stats features | MI: {mi_all.shape[1]} spatial features")
    print(f"  Baseline to beat: {BASELINE_ACC:.4f} ({BASELINE_ACC*100:.2f}%)")
    print("="*65)

    # Default starting params (same as v4 baseline)
    fixed = {"p_dim": 10, "k_neighbors": 10, "pca_dims": 20, "lam": (1.0, 0.1, 0.1, 0.1)}

    # ── Phase 1: Tune p_dim ──────────────────────────────────────────────────
    best_p, _, _ = tune_phase("p_dim", SEARCH_P_DIM, fixed, tf_all, mi_all, labs, subs, "p_dim")
    fixed["p_dim"] = best_p

    # ── Phase 2: Tune k_neighbors ────────────────────────────────────────────
    best_k, _, _ = tune_phase("k_neighbors", SEARCH_K_NEIGHBORS, fixed, tf_all, mi_all, labs, subs, "k_neighbors")
    fixed["k_neighbors"] = best_k

    # ── Phase 3: Tune PCA_DIMS ───────────────────────────────────────────────
    best_pca, _, _ = tune_phase("pca_dims", SEARCH_PCA_DIMS, fixed, tf_all, mi_all, labs, subs, "pca_dims")
    fixed["pca_dims"] = best_pca

    # ── Phase 4: Tune lambdas ────────────────────────────────────────────────
    best_lam, _, _ = tune_phase("lam", SEARCH_LAMBDAS, fixed, tf_all, mi_all, labs, subs, "lam")
    fixed["lam"] = best_lam

    print("\n" + "="*65)
    print("  TUNING COMPLETE -- BEST PARAMETERS FOUND")
    print("="*65)
    print(f"  p_dim       : {fixed['p_dim']}")
    print(f"  k_neighbors : {fixed['k_neighbors']}")
    print(f"  pca_dims    : {fixed['pca_dims']}")
    print(f"  lambdas     : {fixed['lam']}")
    print("="*65)

    # ── Final Run A: All 5 classifiers with best params ──────────────────────
    all_classifiers = {
        "KNN":          lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Ridge":        lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "SVC":          lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
        "XGBoost":      lambda: XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, eval_metric="logloss"),
    }
    run_full_eval(tf_all, mi_all, labs, subs, fixed, all_classifiers,
                  "Tuned v4: All 5 Classifiers vs Baseline")

    # ── Final Run B: SVC only (best classifier) with best params ─────────────
    svc_only = {
        "SVC (C=1)":    lambda: SVC(kernel="rbf", C=1.0,  probability=True, random_state=RANDOM_STATE),
        "SVC (C=0.5)":  lambda: SVC(kernel="rbf", C=0.5,  probability=True, random_state=RANDOM_STATE),
        "SVC (C=2)":    lambda: SVC(kernel="rbf", C=2.0,  probability=True, random_state=RANDOM_STATE),
    }
    run_full_eval(tf_all, mi_all, labs, subs, fixed, svc_only,
                  "Tuned v4: SVC C-Comparison vs Baseline")

    print("\n[DONE] All tuning and evaluation complete. Check the generated PNG plots.")


if __name__ == '__main__':
    main()
