"""
Tuning p_dim only on v4 Baseline (TF-stats + MI)
=================================================
This script ONLY runs Phase 1 (tuning p_dim) and then exits.
This allows your laptop to finish quickly and cool down.
"""

import os
import sys
import warnings
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

N_FOLDS   = 3
K_TF      = 128
GEDLCE_ITER = 30
BASELINE_ACC = 0.6580

SEARCH_P_DIM       = [5, 10, 15, 20, 25, 30]


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


def run_cv(tf_all, mi_all, labs, subs, p_dim, k_neighbors, pca_dims, lam):
    uniq = np.unique(subs)
    slm  = {s: labs[subs == s][0] for s in uniq}
    sl   = np.array([slm[s] for s in uniq])
    cv   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []

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
            g.fit([Xtf_tr.T, Xmi_tr.T], y_train)
        except Exception:
            results.append(0.0)
            continue

        Ft = g.transform([Xtf_tr.T, Xmi_tr.T]).T
        Fe = g.transform([Xtf_te.T, Xmi_te.T]).T

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        clf.fit(Ft, y_train)
        preds = clf.predict(Fe)
        probs = get_probs(clf, Fe)
        _, subj_preds, _ = majority_vote(test_subs, preds, probs)
        _, subj_y        = subject_ground_truth(test_subs, y_test)
        results.append(accuracy_score(subj_y, subj_preds))

    return np.mean(results)


def main():
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
    print("  TUNING PHASE 1 (p_dim) ONLY")
    print(f"  Baseline to beat: {BASELINE_ACC:.4f} ({BASELINE_ACC*100:.2f}%)")
    print("="*65)

    fixed = {"p_dim": 10, "k_neighbors": 10, "pca_dims": 20, "lam": (1.0, 0.1, 0.1, 0.1)}
    best_p = None
    best_acc = 0.0

    for val in SEARCH_P_DIM:
        acc = run_cv(tf_all, mi_all, labs, subs,
                     p_dim=val,
                     k_neighbors=fixed["k_neighbors"],
                     pca_dims=fixed["pca_dims"],
                     lam=fixed["lam"])
        delta = acc - BASELINE_ACC
        flag = "[+]" if delta > 0 else "[-]"
        print(f"  p_dim={val:<10} RF Test Acc={acc:.4f} vs Baseline={delta:+.4f} {flag}")
        if acc > best_acc:
            best_acc = acc
            best_p = val

    print("\n" + "="*65)
    print("  PHASE 1 COMPLETE")
    print(f"  Best p_dim found: {best_p} (Test Acc: {best_acc:.4f})")
    print("="*65)


if __name__ == '__main__':
    main()
