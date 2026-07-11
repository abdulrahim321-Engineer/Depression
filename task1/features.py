import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
import os
import time
from multiprocessing import Pool, cpu_count

def compute_mi_vectorized_all(digitized, H, triu_i, triu_j, bins=10, n_samples=750):
    n_pairs = len(triu_i)
    mi_features = np.zeros(n_pairs, dtype=np.float32)
    
    # Pre-scale to save multiplication inside the loop
    digitized_scaled = digitized * bins
    
    for idx in range(n_pairs):
        i = triu_i[idx]
        j = triu_j[idx]
        joint_state = digitized_scaled[i] + digitized[j]
        counts = np.bincount(joint_state, minlength=100)
        p_xy = counts / n_samples
        p_xy_safe = np.where(p_xy > 0, p_xy, 1.0)
        H_xy = -np.sum(p_xy * np.log2(p_xy_safe))
        mi_features[idx] = H[i] + H[j] - H_xy
        
    return mi_features

def compute_pli_vectorized_all(phase, triu_i, triu_j):
    # Cast to float32 to reduce memory footprint and speed up trigonometric functions
    phase_f32 = phase.astype(np.float32)
    diff = phase_f32[:, None, :] - phase_f32[None, :, :]
    pli_matrix = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
    return pli_matrix[triu_i, triu_j]

def extract_segment_features_wrapper(args):
    seg, idx, total = args
    n_channels, n_samples = seg.shape
    bins = 10
    
    # --- TEMPORAL FEATURES ---
    mean_feat = np.mean(seg, axis=-1)
    max_feat = np.max(seg, axis=-1)
    min_feat = np.min(seg, axis=-1)
    var_feat = np.var(seg, axis=-1)
    skew_feat = skew(seg, axis=-1)
    kurt_feat = kurtosis(seg, axis=-1)
    stats_features = np.column_stack([mean_feat, max_feat, min_feat, var_feat, skew_feat, kurt_feat]).flatten().astype(np.float32)
    
    # --- SPATIAL FEATURES PREPARATION ---
    triu_i, triu_j = np.triu_indices(n_channels, k=1)
    
    # Pre-digitize for MI
    min_val = seg.min(axis=-1, keepdims=True)
    max_val = seg.max(axis=-1, keepdims=True)
    rng = max_val - min_val
    rng[rng == 0] = 1.0
    digitized = ((seg - min_val) / rng * (bins - 1)).astype(np.int32)
    
    counts_individual = np.zeros((n_channels, bins), dtype=np.float32)
    for b in range(bins):
        counts_individual[:, b] = np.sum(digitized == b, axis=-1)
    p_x = counts_individual / n_samples
    p_x_safe = np.where(p_x > 0, p_x, 1.0)
    H = -np.sum(p_x * np.log2(p_x_safe), axis=-1).astype(np.float32)
    
    # 1. Pearson Correlation
    corr_mat = np.corrcoef(seg)
    pearson_features = corr_mat[triu_i, triu_j].astype(np.float32)
    
    # 2. PLI
    analytic = hilbert(seg, axis=-1)
    phase = np.angle(analytic)
    pli_features = compute_pli_vectorized_all(phase, triu_i, triu_j).astype(np.float32)
    
    # 3. MI
    mi_features = compute_mi_vectorized_all(digitized, H, triu_i, triu_j, bins, n_samples)
    
    tf = stats_features
    sf = np.concatenate([pearson_features, pli_features, mi_features])
    
    return idx, tf, sf

def main():
    print("Loading preprocessed data...")
    if not os.path.exists("preprocessed_data.npz"):
        print("Error: preprocessed_data.npz not found.")
        return
        
    preprocessed = np.load("preprocessed_data.npz")
    data = preprocessed['data'] # (N, 128, 750)
    labels = preprocessed['labels']
    subjects = preprocessed['subjects']
    
    n_segments = data.shape[0]
    print(f"Total segments to extract features from: {n_segments}")
    
    # Prepare arguments for multiprocessing
    tasks = [(data[i], i, n_segments) for i in range(n_segments)]
    
    # Use 90% of available CPU cores
    n_cores = max(1, int(cpu_count() * 0.9))
    print(f"Starting multiprocessing pool with {n_cores} workers...")
    
    all_tf = [None] * n_segments
    all_sf = [None] * n_segments
    
    start_time = time.time()
    completed = 0
    
    with Pool(processes=n_cores) as pool:
        for idx, tf, sf in pool.imap_unordered(extract_segment_features_wrapper, tasks, chunksize=5):
            all_tf[idx] = tf
            all_sf[idx] = sf
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (n_segments - completed) / rate
                print(f"Completed {completed}/{n_segments} segments... Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
                
    all_tf = np.array(all_tf)
    all_sf = np.array(all_sf)
    
    # Split spatial features into pearson, pli, mi for the model
    # pearson size: 8128, pli size: 8128, mi size: 8128
    all_pearson = all_sf[:, :8128]
    all_pli = all_sf[:, 8128:16256]
    all_mi = all_sf[:, 16256:]
    
    print("Saving extracted features...")
    np.savez_compressed("extracted_features.npz", 
                        tf=all_tf, 
                        pearson=all_pearson,
                        pli=all_pli,
                        mi=all_mi,
                        labels=labels, 
                        subjects=subjects)
    
    print("Feature extraction completed successfully!")

if __name__ == '__main__':
    main()