"""
Topographical Analysis of EEG Metrics (PE, Skewness, Kurtosis)
Final Degree Project – Bioinformatics

This script computes and visualizes topographical EEG maps using three different metrics:
Permutation Entropy (PE), Skewness, and Kurtosis. The maps are generated for both Eyes Open (EO)
and Eyes Closed (EC) brain states. Statistical comparisons and p-value maps are included to 
highlight significant differences across scalp locations.

Author: Natalia López
Date: June 2025
"""

# ----- 1. Import required libraries -----
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import mne
from scipy import stats

# ----- 2. Load the precomputed EEG metrics for EO and EC conditions -----
# Each matrix has shape (subjects × channels)
pe_eyes_closed = np.load('MATRIX_FINAL_VALUES/EC_pe_raw_4_1_w.npy')
pe_eyes_open = np.load('MATRIX_FINAL_VALUES/EO_pe_raw_4_1_W.npy')

skew_ec = np.load('MATRIX_FINAL_VALUES/EC_skew_raw_4_1_w.npy')
skew_eo = np.load('MATRIX_FINAL_VALUES/EO_skew_raw_4_1_W.npy')

kurt_ec = np.load('MATRIX_FINAL_VALUES/EC_kurt_raw_4_1_w.npy')
kurt_eo = np.load('MATRIX_FINAL_VALUES/EO_kurt_raw_4_1_W.npy')

# ----- 3. Load EEG electrode montage information -----
# A sample EDF file is used to obtain the electrode positions (standard 10-05 system)
edf_path = 'files-2/S001/S001R01.edf'
raw = mne.io.read_raw_edf(edf_path, verbose=None)
raw.load_data()
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")

# ----- 4. Function to compute group-level statistics -----
def compute_stats(eo, ec):
    """
    Compute average values and independent t-test between EO and EC for each channel.
    Returns the average signals, their difference, and the corresponding p-values.
    """
    av_eo = np.mean(eo, axis=0)
    av_ec = np.mean(ec, axis=0)
    t_stats, p_vals = stats.ttest_ind(eo, ec, axis=0, equal_var=False)
    diff = av_eo - av_ec
    return av_eo, av_ec, diff, p_vals

# Compute statistics for each metric
av_pe_eo, av_pe_ec, diff_pe, pval_pe = compute_stats(pe_eyes_open, pe_eyes_closed)
av_skew_eo, av_skew_ec, diff_skew, pval_skew = compute_stats(skew_eo, skew_ec)
av_kurt_eo, av_kurt_ec, diff_kurt, pval_kurt = compute_stats(kurt_eo, kurt_ec)

# ----- 5. Plot topographic maps -----
# General figure settings
plt.rcParams.update({'font.size': 32})  # Make font larger for better readability
fig, axes = plt.subplots(3, 4, figsize=(30, 22))
plt.subplots_adjust(bottom=0.2, hspace=0.5)

# Define the three metrics to display
metrics = [
    ("a)", av_pe_eo, av_pe_ec, diff_pe, pval_pe),
    ("b)", av_skew_eo, av_skew_ec, diff_skew, pval_skew),
    ("c)", av_kurt_eo, av_kurt_ec, diff_kurt, pval_kurt)
]

# Loop through each metric and generate the four topomaps
for i, (label, eo, ec, diff, pval) in enumerate(metrics):
    # Define color ranges for EO/EC and difference
    vmin = min(eo.min(), ec.min())
    vmax = max(eo.max(), ec.max())
    diff_lim = max(abs(diff.min()), abs(diff.max()))

    # EO topomap
    im1, _ = mne.viz.plot_topomap(eo, raw.info, cmap='plasma', contours=0,
                                  image_interp='cubic', vlim=(vmin, vmax), axes=axes[i][0], show=False)
    axes[i][0].set_title(f"{label}   EO", fontsize=32, fontweight='bold')

    # EC topomap
    im2, _ = mne.viz.plot_topomap(ec, raw.info, cmap='plasma', contours=0,
                                  image_interp='cubic', vlim=(vmin, vmax), axes=axes[i][1], show=False)
    axes[i][1].set_title("EC", fontsize=32, fontweight='bold')

    # Difference map (EO - EC)
    im3, _ = mne.viz.plot_topomap(diff, raw.info, cmap='coolwarm', contours=0,
                                  image_interp='cubic', vlim=(-diff_lim, diff_lim), axes=axes[i][2], show=False)
    axes[i][2].set_title("Difference", fontsize=32, fontweight='bold')

    # P-value map (log scale for first row only)
    log_vmin = pval.min()
    log_vmax = pval.max()
    log_ticks = np.round([log_vmin, (log_vmin + log_vmax) / 2, log_vmax], 2)

    if i == 0:
        im4, _ = mne.viz.plot_topomap(pval, raw.info, cmap='plasma', contours=0,
                                      image_interp='linear',
                                      cnorm=colors.LogNorm(vmin=log_vmin, vmax=log_vmax),
                                      axes=axes[i][3], show=False)
    else:
        im4, _ = mne.viz.plot_topomap(pval, raw.info, cmap='plasma', contours=0,
                                      image_interp='linear',
                                      axes=axes[i][3], show=False)
        im4.set_clim(log_vmin, log_vmax)

    axes[i][3].set_title("p-value", fontsize=32, fontweight='bold')

    # ----- 6. Add colorbars to each plot -----
    fig.colorbar(im1, ax=axes[i][0], orientation='horizontal', fraction=0.04, pad=0.1).ax.tick_params(labelsize=28)
    fig.colorbar(im2, ax=axes[i][1], orientation='horizontal', fraction=0.04, pad=0.1).ax.tick_params(labelsize=28)
    fig.colorbar(im3, ax=axes[i][2], orientation='horizontal', fraction=0.04, pad=0.1).ax.tick_params(labelsize=28)
    cbar4 = fig.colorbar(im4, ax=axes[i][3], orientation='horizontal', fraction=0.04, pad=0.1)
    cbar4.ax.tick_params(labelsize=28)
    if i != 0:
        cbar4.set_ticks(log_ticks)

# ----- 7. Save and display figure -----
plt.savefig("topomaps_PE_Skew_Kurtosis.png", dpi=300)
plt.show()
