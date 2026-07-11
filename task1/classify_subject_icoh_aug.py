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
from sklearn.metrics import accuracy_score

N_FOLDS = 3
PCA_DIMS = 20
GEDLCE_P = 10
GEDLCE_ITER = 30
K_BEST_FEATURES = 128

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

def augment_gaussian(X, y, subs, noise_level=0.1, copies=2):
    X_aug, y_aug, subs_aug = [X], [y], [subs]
    for _ in range(copies):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
        subs_aug.append(subs)
    return np.concatenate(X_aug), np.concatenate(y_aug), np.concatenate(subs_aug)

def run_experiment(tf_all, spatial_views, labs, subs, exp_name):
    uniq = np.unique(subs)
    slm = {s: labs[subs == s][0] for s in uniq}
    sl = np.array([slm[s] for s in uniq])
    
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    classifiers_def = {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
        "Ridge": lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "SVC": lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    }
    
    history_train = {clf_name: [] for clf_name in classifiers_def}
    history_test = {clf_name: [] for clf_name in classifiers_def}
    
    for fold, (tri, tei) in enumerate(cv.split(uniq, sl, groups=uniq)):
        tsubs = uniq[tri]
        esubs = uniq[tei]
        
        tm = np.isin(subs, tsubs)
        em = np.isin(subs, esubs)
        
        X1_tr, X1_te = tf_all[tm], tf_all[em]
        y_train, y_test = labs[tm], labs[em]
        train_subs, test_subs = subs[tm], subs[em]
        
        # ANOVA on the 768 stats TF features
        selector = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
        X1t = selector.fit_transform(X1_tr, y_train)
        X1e = selector.transform(X1_te)
        
        s1 = StandardScaler()
        pc = PCA(n_components=min(PCA_DIMS, X1t.shape[0] - 1, X1t.shape[1]), random_state=42)
        s2 = StandardScaler()
        X1t = s2.fit_transform(pc.fit_transform(s1.fit_transform(X1t)))
        X1e = s2.transform(pc.transform(s1.transform(X1e)))
        
        processed_views_train = [X1t]
        processed_views_test = [X1e]
        
        for view_all in spatial_views:
            X2_tr, X2_te = view_all[tm], view_all[em]
            
            # ANOVA on Spatial Features to reduce noise before PCA
            selector_v = SelectKBest(score_func=f_classif, k=min(K_BEST_FEATURES, X2_tr.shape[1]))
            X2_tr = selector_v.fit_transform(X2_tr, y_train)
            X2_te = selector_v.transform(X2_te)
            
            s1_v = StandardScaler()
            pc_v = PCA(n_components=min(10, X2_tr.shape[0] - 1, X2_tr.shape[1]), random_state=42)
            s2_v = StandardScaler()
            X2t = s2_v.fit_transform(pc_v.fit_transform(s1_v.fit_transform(X2_tr)))
            X2e = s2_v.transform(pc_v.transform(s1_v.transform(X2_te)))
            processed_views_train.append(X2t)
            processed_views_test.append(X2e)
            
        augmented_views = []
        for v in processed_views_train:
            v_aug, y_train_aug, train_subs_aug = augment_gaussian(v, y_train, train_subs, noise_level=0.1, copies=1)
            augmented_views.append(v_aug)
            
        y_train = y_train_aug
        train_subs = train_subs_aug
        processed_views_train = augmented_views

        n_views = len(processed_views_train)
        p_dim = min([GEDLCE_P] + [v.shape[1] for v in processed_views_train])
        
        g = GEDLCE(n_views=n_views, p_dim=p_dim, lambda0=1.0, lambda1=0.1, lambda2=0.1,
                   lambda3=0.1, delta=1.0, max_iter=GEDLCE_ITER, tol=1e-6, k_neighbors=10)
        
        try:
            g.fit([v.T for v in processed_views_train], y_train)
        except Exception as ex:
            continue
            
        Ft = g.transform([v.T for v in processed_views_train]).T
        Fe = g.transform([v.T for v in processed_views_test]).T
        Ft = np.nan_to_num(Ft, nan=0.0, posinf=1000.0, neginf=-1000.0)
        Fe = np.nan_to_num(Fe, nan=0.0, posinf=1000.0, neginf=-1000.0)
        Ft = np.clip(Ft, -1e4, 1e4).astype(np.float32)
        Fe = np.clip(Fe, -1e4, 1e4).astype(np.float32)
        
        for clf_name, clf_inst in classifiers_def.items():
            clf = clf_inst()
            clf.fit(Ft, y_train)
            
            preds = clf.predict(Fe)
            probs = clf.predict_proba(Fe)[:, 1] if hasattr(clf, "predict_proba") else preds.astype(float)
            
            _, test_subj_preds, test_subj_probs = majority_vote(test_subs, preds, probs)
            _, test_y_subj = subject_ground_truth(test_subs, y_test)
            acc_test = accuracy_score(test_y_subj, test_subj_preds)
            history_test[clf_name].append(acc_test)
            
            preds_tr = clf.predict(Ft)
            probs_tr = clf.predict_proba(Ft)[:, 1] if hasattr(clf, "predict_proba") else preds_tr.astype(float)
            _, train_subj_preds, _ = majority_vote(train_subs, preds_tr, probs_tr)
            _, train_y_subj = subject_ground_truth(train_subs, y_train)
            acc_train = accuracy_score(train_y_subj, train_subj_preds)
            history_train[clf_name].append(acc_train)
            
    print("\n" + "=" * 60)
    print(f" RESULTS: {exp_name}")
    print("=" * 60)
    print(f"{'Classifier':<15} {'Train Subject Acc':<20} {'Test Subject Acc (Mean ± Std)':<30}")
    print("-" * 70)
    
    clf_names = list(classifiers_def.keys())
    train_accs = [np.mean(history_train[name]) for name in clf_names]
    test_accs = [np.mean(history_test[name]) for name in clf_names]
    test_stds = [np.std(history_test[name]) for name in clf_names]
    
    for i, clf_name in enumerate(clf_names):
        print(f"{clf_name:<15} {train_accs[i] * 100:.2f}%{' ' * 14}{test_accs[i] * 100:.2f}% ± {test_stds[i] * 100:.2f}%")
        
    x = np.arange(len(clf_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_accs, width, label='Train Acc')
    rects2 = ax.bar(x + width/2, test_accs, width, label='Test Acc', yerr=test_stds, capsize=5)
    
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Train vs Test Accuracy\n{exp_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names)
    ax.legend()
    fig.tight_layout()
    plot_filename = exp_name.replace(":", "").replace("+", "").replace(" ", "_")[:50] + "_learning_curve.png"
    plt.savefig(plot_filename)
    print(f"\n[INFO] Saved learning curve plot to {plot_filename}")

def main():
    feat_path = "extracted_features_v3.npz"
    if not os.path.exists(feat_path):
        feat_path = os.path.join("task1", "extracted_features_v3.npz")
    if not os.path.exists(feat_path):
        print(f"[ERROR] {feat_path} not found.")
        sys.exit(1)
        
    feat = np.load(feat_path)
    tf_all = feat["tf"].astype(np.float64)
    # Extract the original 768 Stats features only (first 768 elements)
    tf_orig = np.nan_to_num(tf_all[:, :768], nan=0.0, posinf=0.0, neginf=0.0)
    
    mi_all = np.nan_to_num(feat["mi"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pearson_all = np.nan_to_num(feat["pearson"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pli_all = np.nan_to_num(feat["pli"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    icoh_all = np.nan_to_num(feat["icoh"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    labs = feat["labels"]
    subs = feat["subjects"]
    
    # Run Baseline Control: Original TF + MI
    # run_experiment(tf_orig, [mi_all], labs, subs, "Baseline: TF + MI")

    # Run 1: Original TF + MI + Pearson + ICoh (Replacing PLI with ICoh)
    # run_experiment(tf_orig, [mi_all, pearson_all, icoh_all], labs, subs, "Run 1: TF + MI + Pearson + ICoh")
    
    # Run 2: Original TF + MI + Pearson + PLI + ICoh (Adding ICoh alongside all existing)
    run_experiment(tf_orig, [mi_all, pearson_all, pli_all, icoh_all], labs, subs, "Run 2: TF + MI + Pearson + PLI + ICoh")

if __name__ == '__main__':
    main()

