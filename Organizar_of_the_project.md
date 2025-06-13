# Final Degree Project – EEG Analysis

This repository contains the scripts, data, and figures generated for my final degree project in Bioinformatics. The project focuses on analyzing EEG signals using Permutation Entropy (PE), Spatial Permutation Entropy (SPE), and other statistical metrics to distinguish between eyes open (EO) and eyes closed (EC) brain states.

---

## Repository Structure

### 1. **scripts_different_metrics/**
Contains scripts that analyze EEG signals using different statistical metrics (mean, variance, skewness, kurtosis, MAD, IQR). Each script focuses on one metric.

### 2. **SPE_dificult_conditions/**
Scripts and figures related to the analysis of SPE under more difficult conditions (e.g., fewer electrodes or shorter signals).

### 3. **EEG Analysis Scripts**
- `egg_analysis_2_with_pval.py`: Generates boxplots with statistical comparisons (p-values) for PE, skewness, and kurtosis between EO and EC.
- `comb_boxplots.py`: Combines all boxplots into one figure for easy visualization.
- `boxplot_represent_pval.py`: Visualizes PE, skewness, and kurtosis **without artifacts**, including significance.
- `comb_topomap.py`: Plots topographic maps for different metrics across electrodes.
- `make_figs.m`: MATLAB script for additional figure generation (not used in final figures).
- `ICA_Corrected.py`: Applies ICA artifact removal to EEG data.
- `PSD.py`: Power Spectral Density analysis of EEG recordings.

### 4. **Machine Learning Scripts**
- `RF_feature_select.py`: Feature selection using Random Forest.
- `RF_single_feature.py`: Evaluation of individual features with RF classification.

### 5. **Data Files**
- `spe_summary_by_subject.csv`: Summary of SPE values per subject.
- Numpy `.npy` files stored in subfolders: contain processed SPE, skewness, and kurtosis values.

### 6. **Figures and Documentation**
- `64_channel_sharbrough.png/.pdf`: Electrode layout used in the analyses.
- `RECORDS`, `ANNOTATORS`: Metadata or auxiliary documentation.




## Author

Natalia López López – Bioinformatics student, final degree project in EEG analysis.


