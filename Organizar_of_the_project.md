# Final Degree Project – EEG Signal Analysis

This repository contains all the scripts and data I used for my final degree project in Bioinformatics. The main goal of my project was to analyze EEG signals to compare brain states (eyes open vs. eyes closed) using different statistical and entropy-based methods.

---

## What this project is about

I focused on analyzing EEG data using several metrics like Permutation Entropy (PE), Skewness, Kurtosis, etc. I also looked into how artifacts and the number of electrodes affect the results. The idea was to better understand the brain’s activity in different states and see which features help distinguish them.

---

##  Repository structure 

### `scripts_different_metrics/`

Here I put all the scripts where I analyze the EEG signals using different statistical metrics:

- `egg_analysis_mean.py`: Calculates the mean signal per channel.
- `egg_analysis_var.py`: Computes the variance of the signals.
- `egg_analysis_kurtosis.py`: Works with kurtosis (to see if the signal is peaky or flat).
- `egg_analysis_IQR.py`: Uses interquartile range for robustness.
- `egg_analysis_MAD.py`: Applies Median Absolute Deviation (less sensitive to outliers).
- `egg_analysis_2_with_skew.py`: This one combines PE and skewness in one figure.

### `SPE_dificult_conditions/`

These scripts are more focused on running the SPE (Spatial Permutation Entropy) analysis in more difficult or "realistic" conditions:

- `31_17_17P.py`: Compares results using 31 vs. 17 electrodes and different configurations.
- `boxplot_spatial.py`: Generates boxplots for horizontal/vertical setups.
- `p_value_spatial.py`: Runs the t-tests between EO and EC conditions.
- `tiempo.py`: Explores how results change over time using time windows.

---

## Other important scripts

These are in the main folder and are central to the project:

- `egg_analysis_2_with_pval.py`: Boxplot comparisons for PE, Skewness, and Kurtosis, including p-values.
- `comb_boxplots.py`: Same metrics, but comparing versions with and without artifacts.
- `comb_topomap.py`: Topographic maps of the metrics using MNE.
- `boxplot_represent_pval.py`: Another version of the boxplots, more focused on clean signals.
- `egg_utils_2.py`: Contains the custom EEG class and helper functions.
- `RF_feature_select.py` and `RF_single_feature.py`: Scripts I used for feature selection with Random Forests.
- `ICA_Corrected.py`: Applies ICA to remove artifacts.
- `PSD.py`: Computes Power Spectral Density (frequency analysis).
- `make_figs.m`: A MATLAB script to generate/clean figures.
- `spe_summary_by_subject.csv`: A table summarizing all subjects and their PE/skew/kurt values.

---

## Data info

Some `.npy` and `.edf` files are not uploaded here because they’re too big. But the structure I used is:

- EEG matrices go inside `vectores/`
- Raw EEG files are in `files-2/`



---

## How to use the code

1. Make sure you have Python 3 installed (I used MNE, NumPy, Matplotlib, SciPy...).
2. Place all `.npy` and `.edf` files in the correct folders.
3. Run the scripts depending on what you want to analyze (plots, topomaps, t-tests...).
4. Some scripts use multiprocessing so they might take a while!

---

## Summary of what I analyzed

- Comparison between eyes open and closed using PE, Skewness and Kurtosis.
- Differences with vs. without artifacts.
- Effect of number of electrodes (especially comparing 31 and 17).
- Changes over time in the signal.
- Visualizations with boxplots and topographic maps.


