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

try:
    from task1.gedlce_torch import GEDLCETorch as GEDLCE
except ImportError:
    try:
        from task1.gedlce import GEDLCE
    except ImportError:
        from gedlce import GEDLCE

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

N_FOLDS = 3
PCA_DIMS = 20
GEDLCE_P = 10
GEDLCE_ITER = 30
K_BEST_FEATURES = 128  # Keep only the top 128 features (out of 2688)

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

def compute_metrics(yt, yp, yb):
    a = accuracy_score(yt, yp)
    pr = precision_score(yt, yp, zero_division=0)
    se = recall_score(yt, yp, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        au = roc_auc_score(yt, yb)
    except:
        au = 0.5
    return a, pr, se, sp, au

def main():
    print("=" * 60)
    print("  MI + TF v4 -- ANOVA Feature Selection + Ensembles")
    print("=" * 60)
    
    feat_path = "extracted_features_v2.npz"
    if not os.path.exists(feat_path):
        print(f"[ERROR] {feat_path} not found. Run features_v2.py first.")
        sys.exit(1)
        
    feat = np.load(feat_path)
    tf_all = feat["tf"].astype(np.float64)
    mi_all = feat["mi"].astype(np.float64)
    labs = feat["labels"]
    subs = feat["subjects"]
    
    print(f"Segments:{tf_all.shape[0]}  Original TF:{tf_all.shape[1]}  MI:{mi_all.shape[1]}  Subjects:{np.unique(subs).size}")
    print(f"Applying ANOVA SelectKBest (k={K_BEST_FEATURES}) to reduce TF feature noise...")
    
    uniq = np.unique(subs)
    slm = {s: labs[subs == s][0] for s in uniq}
    sl = np.array([slm[s] for s in uniq])
    
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    classifiers_def = {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Ridge": lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, eval_metric="logloss"),
        "SVC": lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    }
    
    history = {clf_name: {"train_seg": [], "test_seg": [], "train_subj": [], "test_subj": []} for clf_name in classifiers_def}
    
    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]
        esubs = uniq[tei]
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} | Train Subjects:{len(tsubs)} | Test Subjects:{len(esubs)} ---")
        
        tm = np.isin(subs, tsubs)
        em = np.isin(subs, esubs)
        
        X1_tr_raw, X1_te_raw = tf_all[tm], tf_all[em]
        X2t, X2e = mi_all[tm], mi_all[em]
        y_train, y_test = labs[tm], labs[em]
        
        train_subs = subs[tm]
        test_subs = subs[em]
        
        # 1. Feature Selection on TF (fit only on train, transform both)
        selector = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
        X1t = selector.fit_transform(X1_tr_raw, y_train)
        X1e = selector.transform(X1_te_raw)
        
        # Scale -> PCA -> Scale
        nc1 = nc2 = None
        for nm in ["TF", "MI"]:
            Xtr = X1t if nm == "TF" else X2t
            Xte = X1e if nm == "TF" else X2e
            n = min(PCA_DIMS, Xtr.shape[0] - 1, Xtr.shape[1])
            s1 = StandardScaler()
            pc = PCA(n_components=n, random_state=42)
            s2 = StandardScaler()
            Xtr2 = s2.fit_transform(pc.fit_transform(s1.fit_transform(Xtr)))
            Xte2 = s2.transform(pc.transform(s1.transform(Xte)))
            if nm == "TF":
                X1t, X1e, nc1 = Xtr2, Xte2, n
            else:
                X2t, X2e, nc2 = Xtr2, Xte2, n
                
        pd_ = min(GEDLCE_P, nc1, nc2)
        print(f"  PCA: TF={nc1} MI={nc2} | GEDLCE p_dim={pd_}")
        
        g = GEDLCE(n_views=2, p_dim=pd_, lambda0=1.0, lambda1=0.1, lambda2=0.1,
                   lambda3=0.1, delta=1.0, max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=10)
        
        try:
            print("  Fitting GEDLCE...", end=" ", flush=True)
            g.fit([X1t.T, X2t.T], y_train)
            print("OK")
        except Exception as ex:
            print(f"FAILED: {ex}")
            continue
            
        Ft = g.transform([X1t.T, X2t.T]).T
        Fe = g.transform([X1e.T, X2e.T]).T
        
        for clf_name, clf_inst in classifiers_def.items():
            clf = clf_inst()
            clf.fit(Ft, y_train)
            
            # --- Segment Level evaluation ---
            train_seg_preds = clf.predict(Ft)
            test_seg_preds = clf.predict(Fe)
            train_seg_acc = accuracy_score(y_train, train_seg_preds)
            test_seg_acc = accuracy_score(y_test, test_seg_preds)
            
            # --- Subject Level majority vote evaluation ---
            if hasattr(clf, "predict_proba"):
                train_probs = clf.predict_proba(Ft)
                train_seg_probs = train_probs[:, 1] if train_probs.shape[1] == 2 else np.full(len(Ft), float(clf.classes_[0]))
                test_probs = clf.predict_proba(Fe)
                test_seg_probs = test_probs[:, 1] if test_probs.shape[1] == 2 else np.full(len(Fe), float(clf.classes_[0]))
            else:
                if hasattr(clf, "decision_function"):
                    train_seg_probs = clf.decision_function(Ft)
                    test_seg_probs = clf.decision_function(Fe)
                else:
                    train_seg_probs = train_seg_preds.astype(float)
                    test_seg_probs = test_seg_preds.astype(float)
                    
            _, train_subj_preds, train_subj_probs = majority_vote(train_subs, train_seg_preds, train_seg_probs)
            _, train_y_subj = subject_ground_truth(train_subs, y_train)
            
            _, test_subj_preds, test_subj_probs = majority_vote(test_subs, test_seg_preds, test_seg_probs)
            _, test_y_subj = subject_ground_truth(test_subs, y_test)
            
            train_subj_acc = accuracy_score(train_y_subj, train_subj_preds)
            test_subj_acc = accuracy_score(test_y_subj, test_subj_preds)
            
            history[clf_name]["train_seg"].append(train_seg_acc)
            history[clf_name]["test_seg"].append(test_seg_acc)
            history[clf_name]["train_subj"].append(train_subj_acc)
            history[clf_name]["test_subj"].append(test_subj_acc)
            
            print(f"  [{clf_name:<12}] "
                  f"SEG Train/Test: {train_seg_acc:.4f} / {test_seg_acc:.4f} | "
                  f"SUBJ Train/Test: {train_subj_acc:.4f} / {test_subj_acc:.4f}")
                  
    print("\n" + "=" * 70)
    print(f"Final 3-Fold CV Subject-level Metrics (Average over {N_FOLDS} folds)")
    print("=" * 70)
    print(f"{'Classifier':<15} {'Train Subj Acc':<20} {'Test Subj Acc (Mean ± Std)':<30}")
    print("-" * 70)
    for clf_name in classifiers_def:
        tr_subj_mean = np.mean(history[clf_name]["train_subj"])
        te_subj_mean = np.mean(history[clf_name]["test_subj"])
        te_subj_std = np.std(history[clf_name]["test_subj"])
        print(f"{clf_name:<15} {tr_subj_mean:.4f}               {te_subj_mean:.4f} ± {te_subj_std:.4f}")
    print("=" * 70)
    
    # Plotting Learning/Overfitting Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    clf_names = list(classifiers_def.keys())
    x = np.arange(len(clf_names))
    width = 0.35
    
    # Segment level plot
    axes[0].bar(x - width/2, [np.mean(history[name]["train_seg"]) for name in clf_names], width, label='Train Seg', color='#4f81bd')
    axes[0].bar(x + width/2, [np.mean(history[name]["test_seg"]) for name in clf_names], width, label='Test Seg', color='#c0504d')
    axes[0].set_title('Segment-Level Accuracy (Train vs Test)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clf_names)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Subject level plot
    axes[1].bar(x - width/2, [np.mean(history[name]["train_subj"]) for name in clf_names], width, label='Train Subj', color='#9bbb59')
    axes[1].bar(x + width/2, [np.mean(history[name]["test_subj"]) for name in clf_names], width, label='Test Subj', color='#8064a2')
    axes[1].set_title('Subject-Level Accuracy (Train vs Test)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clf_names)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("learning_curves_v4.png", dpi=300)
    print("\nSaved learning curves visualization to learning_curves_v4.png")

if __name__ == '__main__':
    main()
