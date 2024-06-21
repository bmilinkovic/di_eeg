#%%
import os
from glob import glob
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import pymatreader
import matplotlib.pyplot as plt

# Set paths
data_dir = '/Users/borjan/code/di_eeg/data/raw/'
cleaned_dir = '/Users/borjan/code/di_eeg/data/cleaned/'
file = '/Users/borjan/code/di_eeg/data/raw/H0905X.mat'

# Read custom montage
montage = mne.channels.read_custom_montage('/Users/borjan/code/di_eeg/data/nexstim.sfp')

# Process each .mat file in data_dir
# for file in glob(data_dir + '*.mat'):
subject_file = os.path.basename(file)[:5]  # Extract subject ID from file name
subject_cond = os.path.basename(file)[5]  # Extract condition from file name

# Read raw data and preprocess
rawDataStructure = pymatreader.read_mat(file)
info = mne.create_info(ch_names=rawDataStructure['chlocs']['labels'], sfreq=rawDataStructure['srate'], ch_types='eeg')
info.set_montage(montage)
raw = mne.io.RawArray(rawDataStructure['EEG'].T, info=info)

picks = mne.pick_channels_regexp(raw.ch_names, regexp="EEG.")
raw.plot(order=picks, n_channels=len(picks), block=True)

# Interpolate bad channels
raw.interpolate_bads(method=dict(eeg="spline"), reset_bads=True)

# Save the cleaned raw data
cleaned_file_path = os.path.join(cleaned_dir, f"{subject_file}{subject_cond}_cleaned_raw.fif")
raw.save(cleaned_file_path, overwrite=True)

# %%
