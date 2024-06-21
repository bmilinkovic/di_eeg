#%%
import os
from glob import glob
import numpy as np
import scipy.io as sio
import mne
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt

# Set paths
cleaned_dir = '/data/gpfs/projects/punim1761/di_eeg/data/cleaned/'
preprocessed_dir = '/data/gpfs/projects/punim1761/di_eeg/data/preprocessed/dspm/'


# Fetch fsaverage
fs_dir = fetch_fsaverage(verbose=True)
fs_subjects_dir = os.path.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage'
src_downsampled = mne.setup_source_space(subject, spacing='ico4', subjects_dir=fs_subjects_dir, n_jobs=-1)
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Process each cleaned .fif file in cleaned_dir
for file in glob(cleaned_dir + '*.fif'):
    subject_file = os.path.basename(file)[:5]  # Extract subject ID from file name
    subject_cond = os.path.basename(file)[5]  # Extract condition from file name

    # Load the cleaned raw data
    raw = mne.io.read_raw_fif(file, preload=True)

    # Preprocessing steps
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=128)
    raw_notched = raw_filtered.copy().notch_filter(freqs=[50, 100], method='spectrum_fit')
    raw_downsample = raw_notched.copy().resample(sfreq=256)
    raw_avg_ref = raw_downsample.copy().set_eeg_reference(ref_channels='average')
    raw_downsample.set_eeg_reference(projection=True)

    # Compute covariance matrix
    epochs = mne.make_fixed_length_epochs(raw_downsample, duration=2)
    data_cov = mne.compute_covariance(epochs, method='empirical')

    # Compute forward solution
    fwd = mne.make_forward_solution(raw_downsample.info, trans=trans, src=src_downsampled,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)

    # Compute spatial filter
    filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05,
                                        pick_ori='max-power', weight_norm='unit-noise-gain', rank=None)

    # Compute source time course
    stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)
    
    # 1. Visual Inspection
    psd_fig = raw_downsample.plot_psd(fmax=128)  # Plot original PSD up to 150 Hz
    psd_fig.savefig(os.path.join(preprocessed_dir, f"{subject_file}{subject_cond}_psd_lcmv1.png"))

    plt.close('all')

    # Apply parcellation
    labels = mne.read_labels_from_annot(subject, 'HCPMMP1_combined', subjects_dir=fs_subjects_dir)
    label_ts = mne.extract_label_time_course(stc, labels, fwd['src'], mode='mean_flip', return_generator=False, allow_empty=True)

    # Save results
    np.save(os.path.join(preprocessed_dir, f"py/{subject_file}{subject_cond}_source_time_series_lcmv1.npy"), label_ts)
    sio.savemat(os.path.join(preprocessed_dir, f"mat/{subject_file}{subject_cond}_source_time_series_lcmv1.mat"),
                {"source_ts": label_ts})

# %%
