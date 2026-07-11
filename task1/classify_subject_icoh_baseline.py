"""
Baseline ICoh Experiment (v4-style with Tuned Parameters)
==========================================================
EXACT same pipeline as classify_subject_v4.py (Baseline) but:
  1. Spatial feature: ICoh  (instead of MI)
  2. GEDLCE parameters: TUNED values from tune_gedlce_baseline.py
     - p_dim       = 20   (was 10)
     - k_neighbors = 5    (was 10)
     - pca_dims    = 20   (unchanged)
     - lambdas     = (1.0, 0.1, 0.5, 0.5)  (was 1.0,0.1,0.1,0.1)
  3. Primary classifier: SVC(C=2.0) [best found by tuning]
  4. Feature file: extracted_features_v3.npz (has ICoh)
  5. TF features: first 768 columns only (stats-only, same as v4 baseline)

Why ICoh instead of MI?
  ICoh (Imaginary Coherence) mathematically cancels volume conduction artefacts
  (signal smearing from skull), leaving only true long-range brain communication.
  MI captures non-linear dependence but is susceptible to volume conduction noise.
  ICoh is therefore theoretically a cleaner spatial biomarker for depression.

Expected benefit: if ICoh provides a cleaner signal than MI in the same 2-view
pipeline, we should see a test accuracy improvement beyond the 65.80% MI baseline.
"""

import os
import sys
import warnings
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    import torch
    torch.manual_seed(RANDOM_STATE)
except ImportError:
    pass

# GEDLCE import (GPU preferred, CPU fallback)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))
    from gedlce_torch import GEDLCETorch as GEDLCE
    print("[INFO] Using GPU-accelerated GEDLCE (gedlce_torch)")
except ImportError:
    try:
        from gedlce import GEDLCE
        print("[INFO] Using CPU GEDLCE (gedlce)")
    except ImportError:
        from task1.gedlce import GEDLCE
        print("[INFO] Using CPU GEDLCE (task1.gedlce)")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
#  TUNED HYPERPARAMETERS  (from tune_gedlce_baseline.py)
# ─────────────────────────────────────────────
N_FOLDS         = 3
K_BEST_TF       = 128       # ANOVA top-k for TF (same as v4 baseline)
PCA_DIMS        = 20        # per-view PCA dims
GEDLCE_P        = 20        # TUNED (was 10)
GEDLCE_K        = 5         # TUNED (was 10)
GEDLCE_ITER     = 30
LAM0, LAM1, LAM2, LAM3 = 1.0, 0.1, 0.5, 0.5   # TUNED

BASELINE_REFERENCE = 0.6580  # Original untuned MI+TF baseline accuracy

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BASELINE ICoh EXPERIMENT (v4 pipeline + Tuned Params)")
    print("  Spatial: ICoh  |  TF: 768 stats  |  ANOVA top-128")
    print(f"  GEDLCE: p_dim={GEDLCE_P}, k={GEDLCE_K}, "
          f"lam=({LAM0},{LAM1},{LAM2},{LAM3})")
    print(f"  Baseline to beat: {BASELINE_REFERENCE:.4f} (65.80%)")
    print("=" * 70)

    # ── Load data ──
    feat_path = "extracted_features_v3.npz"
    if not os.path.exists(feat_path):
        feat_path = os.path.join("task1", "extracted_features_v3.npz")
    if not os.path.exists(feat_path):
        print("[ERROR] extracted_features_v3.npz not found.")
        sys.exit(1)

    feat = np.load(feat_path)

    # TF: first 768 columns = statistical features only (same as v4 baseline)
    tf_all = feat["tf"].astype(np.float64)[:, :768]
    tf_all = np.nan_to_num(tf_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Spatial: ICoh only (no ANOVA — PCA preserves global connectivity structure)
    icoh_all = feat["icoh"].astype(np.float64)
    icoh_all = np.nan_to_num(icoh_all, nan=0.0, posinf=0.0, neginf=0.0)

    labs = feat["labels"]
    subs = feat["subjects"]

    print(f"[INFO] Segments:{tf_all.shape[0]}  TF stats:{tf_all.shape[1]}"
          f"  ICoh:{icoh_all.shape[1]}  Subjects:{np.unique(subs).size}")

    # ── Classifiers ──
    classifiers_def = {
        "KNN":          lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
        "Ridge":        lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "SVC":          lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        "SVC(C=2)":     lambda: SVC(kernel="rbf", C=2.0, probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":      lambda: XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
    }

    history_train = {n: [] for n in classifiers_def}
    history_test  = {n: [] for n in classifiers_def}

    uniq = np.unique(subs)
    slm  = {s: labs[subs == s][0] for s in uniq}
    sl   = np.array([slm[s] for s in uniq])

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]
        esubs = uniq[tei]
        print(f"\n--- Fold {fold+1}/{N_FOLDS} | Train:{len(tsubs)} subjects | Test:{len(esubs)} subjects ---")

        tm = np.isin(subs, tsubs)
        em = np.isin(subs, esubs)

        X_tf_tr, X_tf_te   = tf_all[tm],   tf_all[em]
        X_ic_tr, X_ic_te   = icoh_all[tm], icoh_all[em]
        y_train,  y_test   = labs[tm],     labs[em]
        train_subs = subs[tm]
        test_subs  = subs[em]

        # ── View 1: TF (stats) — ANOVA → Scale → PCA → Scale ──
        sel = SelectKBest(score_func=f_classif, k=min(K_BEST_TF, X_tf_tr.shape[1]))
        X_tf_tr = sel.fit_transform(X_tf_tr, y_train)
        X_tf_te = sel.transform(X_tf_te)

        nc_tf = min(PCA_DIMS, X_tf_tr.shape[0] - 1, X_tf_tr.shape[1])
        s1, pc1, s2 = StandardScaler(), PCA(n_components=nc_tf, random_state=42), StandardScaler()
        X_tf_tr = s2.fit_transform(pc1.fit_transform(s1.fit_transform(X_tf_tr)))
        X_tf_te = s2.transform(pc1.transform(s1.transform(X_tf_te)))

        # ── View 2: ICoh — Scale → PCA → Scale (no ANOVA: preserve global structure) ──
        nc_ic = min(PCA_DIMS, X_ic_tr.shape[0] - 1, X_ic_tr.shape[1])
        s3, pc2, s4 = StandardScaler(), PCA(n_components=nc_ic, random_state=42), StandardScaler()
        X_ic_tr = s4.fit_transform(pc2.fit_transform(s3.fit_transform(X_ic_tr)))
        X_ic_te = s4.transform(pc2.transform(s3.transform(X_ic_te)))

        pd_ = min(GEDLCE_P, nc_tf, nc_ic)
        print(f"  PCA dims: TF={nc_tf}, ICoh={nc_ic} | GEDLCE p_dim={pd_}")

        # ── GEDLCE fusion (tuned params) ──
        g = GEDLCE(
            n_views=2, p_dim=pd_,
            lambda0=LAM0, lambda1=LAM1, lambda2=LAM2, lambda3=LAM3,
            delta=1.0, max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=GEDLCE_K
        )

        print("  Fitting GEDLCE...", end=" ", flush=True)
        try:
            g.fit([X_tf_tr.T, X_ic_tr.T], y_train)
            print("OK")
        except Exception as ex:
            print(f"FAILED: {ex}")
            continue

        Ft = g.transform([X_tf_tr.T, X_ic_tr.T]).T
        Fe = g.transform([X_tf_te.T, X_ic_te.T]).T
        Ft = np.nan_to_num(np.clip(Ft, -1e4, 1e4), nan=0.0).astype(np.float32)
        Fe = np.nan_to_num(np.clip(Fe, -1e4, 1e4), nan=0.0).astype(np.float32)

        # ── Classify ──
        for clf_name, clf_inst in classifiers_def.items():
            clf = clf_inst()
            clf.fit(Ft, y_train)

            preds_tr = clf.predict(Ft)
            probs_tr = get_probs(clf, Ft)
            _, tr_subj_preds, _ = majority_vote(train_subs, preds_tr, probs_tr)
            _, tr_y_subj       = subject_ground_truth(train_subs, y_train)
            acc_train = accuracy_score(tr_y_subj, tr_subj_preds)

            preds_te = clf.predict(Fe)
            probs_te = get_probs(clf, Fe)
            _, te_subj_preds, _ = majority_vote(test_subs, preds_te, probs_te)
            _, te_y_subj        = subject_ground_truth(test_subs, y_test)
            acc_test = accuracy_score(te_y_subj, te_subj_preds)

            history_train[clf_name].append(acc_train)
            history_test[clf_name].append(acc_test)

            gap   = acc_train - acc_test
            delta = acc_test - BASELINE_REFERENCE
            sign  = "[+]" if delta >= 0 else "[-]"
            print(f"  [{clf_name:<12}] SUBJ Train={acc_train:.4f} Test={acc_test:.4f} "
                  f"Gap={gap:+.4f}  vs Baseline={delta:+.4f} {sign}")

    # ─────────────────────────────────────────
    #  Final Summary
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY -- Baseline ICoh (v4 + Tuned Params)")
    print(f"  BASELINE MI+TF (untuned) = {BASELINE_REFERENCE}")
    print("=" * 80)
    print(f"{'Classifier':<14} {'Train Subj':>12} {'Test Subj':>12} {'Gap':>8} {'vs Baseline':>14}")
    print("-" * 68)

    best_clf, best_acc = None, 0.0
    results = {}
    for clf_name in classifiers_def:
        tr = np.mean(history_train[clf_name])
        te = np.mean(history_test[clf_name])
        gap   = tr - te
        delta = te - BASELINE_REFERENCE
        sign  = "[+]" if delta >= 0 else "[-]"
        results[clf_name] = (tr, te, gap, delta)
        print(f"{clf_name:<14} {tr:>12.4f} {te:>12.4f} {gap:>+8.4f} {delta:>+12.4f} {sign}")
        if te > best_acc:
            best_acc, best_clf = te, clf_name

    print("=" * 80)
    print(f"  [*] Best: {best_clf}  |  {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("=" * 80)

    # ─────────────────────────────────────────
    #  Plot
    # ─────────────────────────────────────────
    clf_names  = list(classifiers_def.keys())
    train_vals = [results[n][0] for n in clf_names]
    test_vals  = [results[n][1] for n in clf_names]

    x     = np.arange(len(clf_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_vals, width, label="Train Subj", color="#4f81bd")
    ax.bar(x + width/2, test_vals,  width, label="Test Subj",  color="#c0504d")
    ax.axhline(BASELINE_REFERENCE, linestyle="--", color="green", linewidth=1.8,
               label=f"Baseline MI+TF ({BASELINE_REFERENCE*100:.1f}%)")
    ax.set_title("Baseline ICoh Experiment (v4 + Tuned Params)\nTF Stats-only (ANOVA) + ICoh (PCA)")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=20)
    ax.set_ylabel("Subject-Level Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = "Baseline_ICoh_Tuned_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[INFO] Saved plot -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
