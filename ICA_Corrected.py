import matplotlib.pyplot as plt
#from egg_utils_2 import eeg
import numpy as np
import mne 

# Load data
name ='/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2/S001/S001R01.edf'
raw = mne.io.read_raw_edf(name,verbose=None)
# para ver los sujetos tanto el 109 como el 97
raw.plot()
raw.load_data()
###########################################

# Set montage
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")
###########################################

artifact_picks = mne.pick_channels(raw.ch_names, []) #This line is only needded for plots
#raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# Filter very low frequencies
filt_raw = raw.copy().filter(l_freq=1, h_freq=None)
###########################################

# ICA decompossition
ica = mne.preprocessing.ICA(n_components=5, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica.plot_sources(filt_raw)
ica.plot_components()
###########################################

# Select independent compoments to remove
ica.exclude = [0]
###########################################

# Reconstruct original data without artifacts
reconst_raw = raw.copy()
ica.apply(reconst_raw)
###########################################

# Save reconstructed data with out artifacts
#fname = name[:-4]+'_wo_artifacts.edf'
#mne.export.export_raw(fname, reconst_raw, overwrite=True)
###########################################

# Plots original and reconstructed data (too many channels to properly inspect)
'''raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)'''
###########################################

# In these lines we can load the file that we just saved, do ICA and check
# that the are no artifacts
'''raw = mne.io.read_raw_edf(fname,verbose=None)
raw.load_data()
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")
filt_raw = raw.copy().filter(l_freq=1, h_freq=None)
ica = mne.preprocessing.ICA(n_components=16, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica.plot_sources(filt_raw)
ica.plot_components()'''
###########################################
