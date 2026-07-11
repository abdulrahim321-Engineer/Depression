import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, welch
from scipy.stats import skew, kurtosis
import os
import time
from multiprocessing import Pool, cpu_count

BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
SFREQ = 250
N_SAMPLES = 750
NPERSEG = min(256, N_SAMPLES)

def compute_psd_de_vectorized(seg):
    n_ch = seg.shape[0]
    freqs, Pxx = welch(seg, fs=SFREQ, nperseg=NPERSEG, axis=-1)
    psd_bands = np.zeros((n_ch, 5), dtype=np.float32)
    for b_idx, (lo, hi) in enumerate(BANDS.values()):
        mask = (freqs >= lo) & (freqs < hi)
        psd_bands[:, b_idx] = Pxx[:, mask].mean(axis=-1) if mask.any() else 1e-10
    psd_log = np.log(psd_bands + 1e-10)
    bw = np.array([b[1]-b[0] for b in BANDS.values()], dtype=np.float32)
    band_var = psd_bands * bw[np.newaxis, :]
    de = 0.5 * np.log(2*np.pi*np.e*(band_var+1e-10))
    return psd_log.flatten().astype(np.float32), de.flatten().astype(np.float32)

def compute_entropy_vectorized(seg):
    n_ch = seg.shape[0]
    freqs, Pxx = welch(seg, fs=SFREQ, nperseg=NPERSEG, axis=-1)
    Pxx_norm = Pxx / (Pxx.sum(axis=-1, keepdims=True) + 1e-10)
    sp_ent = -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-10), axis=-1)
    diff1 = np.diff(seg, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    std1 = diff1.std(axis=-1) + 1e-10
    std0 = seg.std(axis=-1) + 1e-10
    samp_ent_approx = np.log(std1 / std0)
    return np.column_stack([sp_ent, samp_ent_approx]).flatten().astype(np.float32)

def compute_hjorth_vectorized(seg):
    d1 = np.diff(seg, axis=-1)
    d2 = np.diff(d1, axis=-1)
    activity = seg.var(axis=-1)
    mob_seg = d1.var(axis=-1) / (activity + 1e-10)
    mob_d1 = d2.var(axis=-1) / (d1.var(axis=-1) + 1e-10)
    mobility = np.sqrt(mob_seg)
    complexity = np.sqrt(mob_d1) / (mobility + 1e-10)
    return np.column_stack([activity, mobility, complexity]).flatten().astype(np.float32)

def compute_mi_vectorized_all(digitized, H, triu_i, triu_j, bins=10, n_samples=750):
    n_pairs = len(triu_i)
    mi_features = np.zeros(n_pairs, dtype=np.float32)
    digitized_scaled = digitized * bins
    for idx in range(n_pairs):
        i, j = triu_i[idx], triu_j[idx]
        joint_state = digitized_scaled[i] + digitized[j]
        counts = np.bincount(joint_state, minlength=100)
        p_xy = counts / n_samples
        p_xy_safe = np.where(p_xy > 0, p_xy, 1.0)
        H_xy = -np.sum(p_xy * np.log2(p_xy_safe))
        mi_features[idx] = H[i] + H[j] - H_xy
    return mi_features

def compute_pli_vectorized_all(phase, triu_i, triu_j):
    phase_f32 = phase.astype(np.float32)
    diff = phase_f32[:, None, :] - phase_f32[None, :, :]
    pli_matrix = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
    return pli_matrix[triu_i, triu_j]

def extract_segment_features_wrapper(args):
    seg, idx, total = args
    n_channels, n_samples = seg.shape
    bins = 10
    mean_feat = np.mean(seg, axis=-1); max_feat = np.max(seg, axis=-1)
    min_feat = np.min(seg, axis=-1); var_feat = np.var(seg, axis=-1)
    skew_feat = skew(seg, axis=-1); kurt_feat = kurtosis(seg, axis=-1)
    stats_features = np.column_stack([mean_feat,max_feat,min_feat,var_feat,skew_feat,kurt_feat]).flatten().astype(np.float32)
    psd_feat, de_feat = compute_psd_de_vectorized(seg)
    ent_feat = compute_entropy_vectorized(seg)
    hjorth_feat = compute_hjorth_vectorized(seg)
    tf = np.concatenate([stats_features, psd_feat, de_feat, ent_feat, hjorth_feat])
    triu_i, triu_j = np.triu_indices(n_channels, k=1)
    min_val = seg.min(axis=-1, keepdims=True); max_val = seg.max(axis=-1, keepdims=True)
    rng = max_val - min_val; rng[rng == 0] = 1.0
    digitized = ((seg - min_val) / rng * (bins - 1)).astype(np.int32)
    counts_individual = np.zeros((n_channels, bins), dtype=np.float32)
    for b in range(bins):
        counts_individual[:, b] = np.sum(digitized == b, axis=-1)
    p_x = counts_individual / n_samples
    p_x_safe = np.where(p_x > 0, p_x, 1.0)
    H = -np.sum(p_x * np.log2(p_x_safe), axis=-1).astype(np.float32)
    corr_mat = np.corrcoef(seg)
    pearson_features = corr_mat[triu_i, triu_j].astype(np.float32)
    analytic = hilbert(seg, axis=-1); phase = np.angle(analytic)
    pli_features = compute_pli_vectorized_all(phase, triu_i, triu_j).astype(np.float32)
    mi_features = compute_mi_vectorized_all(digitized, H, triu_i, triu_j, bins, n_samples)
    sf = np.concatenate([pearson_features, pli_features, mi_features])
    return idx, tf, sf

def main():
    print('Loading preprocessed data...')
    data_path = 'preprocessed_data.npz'
    if not os.path.exists(data_path):
        print('Error: preprocessed_data.npz not found.'); return
    preprocessed = np.load(data_path)
    data = preprocessed['data']; labels = preprocessed['labels']; subjects = preprocessed['subjects']
    n_segments = data.shape[0]; n_ch = 128; n_sf = (n_ch*(n_ch-1))//2
    print(f'Total segments: {n_segments}')
    n_tf = n_ch*6 + n_ch*5 + n_ch*5 + n_ch*2 + n_ch*3
    print(f'Expected TF dim: {n_tf} (stats=768, PSD=640, DE=640, Ent=256, Hjorth=384)')
    tasks = [(data[i], i, n_segments) for i in range(n_segments)]
    n_cores = max(1, int(cpu_count() * 0.9))
    print(f'Starting pool with {n_cores} workers...')
    all_tf = [None]*n_segments; all_sf = [None]*n_segments
    start_time = time.time(); completed = 0
    with Pool(processes=n_cores) as pool:
        for idx, tf, sf in pool.imap_unordered(extract_segment_features_wrapper, tasks, chunksize=5):
            all_tf[idx] = tf; all_sf[idx] = sf; completed += 1
            if completed % 100 == 0:
                elapsed = time.time()-start_time; rate = completed/elapsed
                remaining = (n_segments-completed)/rate
                print(f'  {completed}/{n_segments}  elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s')
    all_tf = np.array(all_tf); all_sf = np.array(all_sf)
    all_pearson = all_sf[:, :n_sf]; all_pli = all_sf[:, n_sf:2*n_sf]; all_mi = all_sf[:, 2*n_sf:]
    out_path = 'extracted_features_v2.npz'
    print(f'Saving to {out_path}...')
    np.savez_compressed(out_path, tf=all_tf, pearson=all_pearson, pli=all_pli, mi=all_mi, labels=labels, subjects=subjects)
    print('Done! TF shape:', all_tf.shape, ' MI shape:', all_mi.shape)

if __name__ == '__main__':
    main()
