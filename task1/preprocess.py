import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import mne

def preprocess_subject(mat_path, label, subject_id):
    print(f"Preprocessing Subject {subject_id} (label={label}) from {mat_path}...")
    # Load mat file
    mat = sio.loadmat(mat_path)
    # Find the variable name containing the EEG data
    data_key = None
    for k in mat.keys():
        if k.startswith('a') and k.endswith('mat'):
            data_key = k
            break
    if data_key is None:
        # fallback to find any 2D array that might be the EEG data
        for k, v in mat.items():
            if not k.startswith('__') and isinstance(v, np.ndarray) and len(v.shape) == 2 and v.shape[0] >= 128:
                data_key = k
                break
    
    if data_key is None:
        raise ValueError(f"Could not find EEG data variable in {mat_path}")
        
    data = mat[data_key][:128, :] # Keep first 128 channels
    sfreq = float(mat['samplingRate'][0, 0])
    
    # Create MNE raw array
    ch_names = [f"E{i}" for i in range(1, 129)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Bandpass filter 1-40 Hz
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
    
    # ICA artifact removal
    # We use 20 components. Eyeblink artifacts are typically captured in components 
    # that correlate highly with frontal channels.
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter='auto', verbose=False)
    ica.fit(raw, verbose=False)
    
    # Find components to exclude (those highly correlated with Fp1/Fp2, i.e., E22 and E9)
    # 0-indexed: E22 is index 21, E9 is index 8
    fp1_idx = 21
    fp2_idx = 8
    
    # We compute correlation of ICA source signals with the raw prefrontal channels
    sources = ica.get_sources(raw).get_data()
    raw_data = raw.get_data()
    
    corrs_fp1 = [abs(np.corrcoef(sources[i], raw_data[fp1_idx])[0, 1]) for i in range(sources.shape[0])]
    corrs_fp2 = [abs(np.corrcoef(sources[i], raw_data[fp2_idx])[0, 1]) for i in range(sources.shape[0])]
    
    # Exclude components that have a correlation higher than 0.35 with either frontal channel
    exclude_idx = []
    for i in range(sources.shape[0]):
        if corrs_fp1[i] > 0.35 or corrs_fp2[i] > 0.35:
            exclude_idx.append(i)
            
    # Always exclude at least the top correlated component if none crossed the threshold
    if not exclude_idx:
        max_corr_idx = np.argmax([max(c1, c2) for c1, c2 in zip(corrs_fp1, corrs_fp2)])
        exclude_idx = [int(max_corr_idx)]
        
    ica.exclude = exclude_idx
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose=False)
    
    # Extract middle 120 seconds
    # MODMA resting state is 5 minutes (300 seconds). Middle 120s is from 90s to 210s.
    start_sec = 90.0
    end_sec = 210.0
    
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)
    
    cleaned_data = raw_cleaned.get_data()
    # Check if we have enough samples
    if cleaned_data.shape[1] < end_sample:
        # If recording is shorter, take the middle of the available data
        total_len_sec = cleaned_data.shape[1] / sfreq
        mid_sec = total_len_sec / 2.0
        start_sec = max(0.0, mid_sec - 60.0)
        end_sec = min(total_len_sec, mid_sec + 60.0)
        start_sample = int(start_sec * sfreq)
        end_sample = int(end_sec * sfreq)
        
    middle_data = cleaned_data[:, start_sample:end_sample]
    
    # Segment into 3-second non-overlapping windows (750 samples each at 250Hz)
    win_len = int(3.0 * sfreq) # 750 samples
    n_segments = middle_data.shape[1] // win_len
    
    segments = []
    for i in range(n_segments):
        seg = middle_data[:, i*win_len:(i+1)*win_len]
        segments.append(seg)
        
    segments = np.array(segments) # Shape: (n_segments, 128, 750)
    print(f"Generated {len(segments)} segments for subject {subject_id}")
    return segments

def main():
    dataset_dir = "EEG_128channels_resting_lanzhou_2015"
    info_path = os.path.join(dataset_dir, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    
    # Read subject info
    df_info = pd.read_excel(info_path)
    # Map subject ID to label (MDD = 1, HC = 0)
    subject_labels = {}
    for idx, row in df_info.iterrows():
        sub_id = str(row['subject id']).strip()
        label = 1 if row['type'].strip().upper() == 'MDD' else 0
        subject_labels[sub_id] = label
        
    mat_files = glob.glob(os.path.join(dataset_dir, "*.mat"))
    
    all_segments = []
    all_labels = []
    all_subject_ids = []
    
    for mat_path in mat_files:
        filename = os.path.basename(mat_path)
        # The filename typically starts with '0' + subject_id
        # e.g., '02010002rest 20150416 1017..mat' -> subject_id '2010002'
        # Let's extract digits
        digits = "".join([c for c in filename if c.isdigit()])
        # Normally the subject ID has 7 digits (e.g. 2010002, 2020008, etc.)
        # Filename digits might have date too, so let's match the known subject IDs from excel
        subject_id = None
        for k in subject_labels.keys():
            if k in digits:
                subject_id = k
                break
                
        if subject_id is None:
            print(f"Warning: Could not match filename {filename} to any subject ID. Skipping.")
            continue
            
        label = subject_labels[subject_id]
        
        try:
            segments = preprocess_subject(mat_path, label, subject_id)
            all_segments.append(segments)
            all_labels.extend([label] * len(segments))
            all_subject_ids.extend([subject_id] * len(segments))
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            
    # Concatenate all segments
    all_segments = np.concatenate(all_segments, axis=0) # Shape: (Total_Segments, 128, 750)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    
    # Save preprocessed data
    np.savez_compressed("preprocessed_data.npz", 
                        data=all_segments, 
                        labels=all_labels, 
                        subjects=all_subject_ids)
    print("Preprocessing completed!")
    print(f"Total segments: {all_segments.shape[0]}")
    print(f"Data shape: {all_segments.shape}")

if __name__ == '__main__':
    main()
