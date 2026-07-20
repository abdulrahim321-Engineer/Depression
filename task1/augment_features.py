"""
Data Augmentation: Feature-Level Gaussian Noise (2x)
=====================================================
Reads extracted_features_v3.npz and doubles the dataset by creating
one augmented copy of every subject's segments (Gaussian noise injection).

Output: extracted_features_v3_aug.npz  (2x the original size)
        extracted_features_v2_aug.npz  (2x, for v4 baseline experiments)

Strategy:
  For each subject, ALL of their segments get a noisy copy:
    augmented_segment = original_segment + N(0, noise_std * std_per_feature)
  Augmented subjects get a new subject ID = original_id + 1000 (avoids collision).

Run this ONCE on Kaggle GPU. The output NPZ files are then used by all experiments.
No GPU needed for this script — pure numpy. Fast (~30 seconds).
"""

import os
import sys
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

NOISE_FRACTION = 0.02     # noise = 2% of each feature's std across the dataset
COPIES         = 1        # 1 copy = 2x dataset; set to 2 for 3x


def augment_array(X, subs, noise_fraction=NOISE_FRACTION, copies=COPIES):
    """
    For each subject, create `copies` noisy versions.
    Returns augmented X, augmented subject IDs (original IDs + 1000*copy_idx).
    """
    per_feat_std = X.std(axis=0)  # std across all segments for each feature
    per_feat_std = np.where(per_feat_std == 0, 1e-8, per_feat_std)

    aug_X    = [X]
    aug_subs = [subs]

    for c in range(1, copies + 1):
        noise = np.random.normal(0, noise_fraction * per_feat_std, size=X.shape)
        aug_X.append(X + noise)
        if np.issubdtype(subs.dtype, np.number):
            aug_subs.append(subs + 1000 * c)   # new unique numeric subject IDs
        else:
            aug_subs.append(np.array([f"{s}_aug{c}" for s in subs])) # new unique string subject IDs

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_subs, axis=0)


def augment_file(src_path, dst_path, copies=COPIES):
    print(f"\n[AUG] Loading {src_path} ...")
    feat = np.load(src_path)

    labs = feat["labels"]
    subs = feat["subjects"]

    keys_to_augment = [k for k in feat.files if k not in ("labels", "subjects")]
    print(f"[AUG] Arrays to augment: {keys_to_augment}")
    print(f"[AUG] Original: {labs.shape[0]} segments | {np.unique(subs).size} subjects")

    # Augment the first array (TF) to get shared new_subs/new_labs
    first_key  = keys_to_augment[0]
    _, new_subs = augment_array(feat[first_key].astype(np.float64), subs, copies=copies)

    # Build new labels (same label for augmented subject copies)
    subj_label_map = {s: labs[subs == s][0] for s in np.unique(subs)}
    # Augmented subject IDs share the same label as their original
    orig_subj_ids = np.unique(subs)
    aug_label_parts = [labs]
    for c in range(1, copies + 1):
        aug_labs_c = np.array([subj_label_map[s] for s in subs])
        aug_label_parts.append(aug_labs_c)
    new_labs = np.concatenate(aug_label_parts, axis=0)

    # Augment each feature array
    save_dict = {"labels": new_labs, "subjects": new_subs}
    for key in keys_to_augment:
        arr = feat[key].astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        new_arr, _ = augment_array(arr, subs, copies=copies)
        save_dict[key] = new_arr.astype(np.float32)   # float32 to save disk space
        print(f"[AUG]   {key}: {arr.shape} -> {new_arr.shape}")

    print(f"[AUG] New dataset: {new_labs.shape[0]} segments | {np.unique(new_subs).size} subjects")
    np.savez_compressed(dst_path, **save_dict)
    print(f"[AUG] Saved -> {dst_path}")


def main():
    # ── Paths (works on both local and Kaggle) ───────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # On Kaggle, dataset files are at /kaggle/input/<dataset-name>/
    kaggle_input = "/kaggle/input/eeg-depression-features"

    def find_file(filename):
        root_dir = os.path.dirname(base_dir)
        candidates = [
            os.path.join(root_dir, "extracted_datasets", filename),
            os.path.join(base_dir, filename),
            os.path.join(kaggle_input, filename),
        ]
        import glob
        if os.path.exists("/kaggle/input"):
            candidates.extend(glob.glob(f"/kaggle/input/**/{filename}", recursive=True))
            
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"{filename} not found in any expected location: {candidates}")

    def out_path(filename):
        """Output goes to task1/ locally, or /kaggle/working/ on Kaggle."""
        if os.path.exists("/kaggle/working"):
            return os.path.join("/kaggle/working", filename)
        return os.path.join(base_dir, "task1", filename)

    # ── Augment v3 (used for ICoh/multi-view experiments) ────────────────────
    v3_src = find_file("extracted_features_v3.npz")
    v3_dst = out_path("extracted_features_v3_aug.npz")
    augment_file(v3_src, v3_dst, copies=COPIES)

    # ── Augment v2 (used for v4 baseline) ────────────────────────────────────
    v2_src = find_file("extracted_features_v2.npz")
    v2_dst = out_path("extracted_features_v2_aug.npz")
    augment_file(v2_src, v2_dst, copies=COPIES)

    print("\n[DONE] Both augmented datasets saved successfully.")
    print(f"  v2 aug -> {v2_dst}")
    print(f"  v3 aug -> {v3_dst}")


if __name__ == "__main__":
    main()
