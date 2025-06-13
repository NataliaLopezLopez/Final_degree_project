# eeg-spatial-analysis

Python scrips for the spatial analysis of EEG signals. 
These scrips reproduce the analysis performed and published in the work: ["Permutation entropy analysis of EEG signals for distinguishing eyes-open and eyes-closed brain states: Comparison of different approaches"](https://doi.org/10.1063/5.0200029).

## Scripts

### Summary of the scripts included:

eeg_analysis.py: Main analysis 

eeg_utils.py: Supporting functions for eeg_analysis.py

RF_single_feature.py: Random forest classification of the data

RF_feature_select.py: Performs feature selection before running the RF classification (allows multiple feature classification)

make_figs.m: Matlab script that outputs the figures of Gancio et al. (2024)

## eeg_analysis.py
This script performs the main analysis publish in Gancio et al. (2024). 
The following parameters have to be set:

`number_of_subjects`: shouldn't be larger that 108, dataset has 109 but subject 109 has some not valid values at the end

`filt_mode`:'raw' or 'filt', for considering only the alpha band

`word_length`: Word length of the symbols in the Ordinal Pattern analysis. For a 64 channel EEG, we recommend only set it word_length=3

`lag`: Spatial lag of the symbols in the Ordinal Pattern analysis. For a 64 channel EEG, we recommend only set it lag=1

`experiment_run`:Run of the experiment, 1 for Eyes Open, and 2 to Eyes Closed

'analysis_mode': Controls the symbol construction (temporal or spatial) and how the averages are performed.

'ensemble' : spatial analysis, but for each time, the mean value of SPE in all subjects if provided. This analysis corresponds to the one of Boaretto et al. (2023), and outputs the data used to plot Fig.4 of Gancio et al. (2024). It outputs SPE for different arrangements:

	linear: straight forward, as come from the dataset, ordering (first approach of Boaretto et al. (2023))

	boaretto's best: best arrangement of electrodes find by Boaretto et al. (2023)

	horizontal symbols

	vertical symbols


'spatial': Spatial analysis as performed by Gancio et al. (2024), an average in time of the SPE values is provided (one averaged quantity for each subject). This provided the data for Fig. 5, and partial of Figs. 7 and 8 of  Gancio et al. (2024).

'temporal':  Temporal analysis as performed by Gancio et al. (2024), an average in space of the Permutation Entropy (PE) values is provided (one averaged quantity for each subject). This provided the data for Fig. 6, and the additional data for Figs. 7 and 8 of  Gancio et al. (2024).

## RF_single_feature.py

Random Forest (RF) classification of the files that `eeg_analysis.py` outputs. Two file paths have to be specified, corresponding to the two different categories of the classification. These files have to be a column .csv file, and contain the same number of subjects (which has to also be set manually to the variable `number_of_subjects`. The files should correspond to the same pre-processing and analysis of the `eeg_analysis.py`, but different runs of the experiment, to the RF classification to make sense.

The parameters of the k-fold cross validation can also be set.

## RF_feature_select.py

RF classifier with a feature selection stage. Script loads a file with all the features, and performs a feature selection that can be 'forward' or 'backward', until the 'n_features_to_select' has been reached. Then the script performs the classification and reports results.


