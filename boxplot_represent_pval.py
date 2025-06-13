# Libraries for statistical analysis, data handling, and plotting
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Function to create a boxplot in a given subplot (ax)
# ---------------------------------------------------------------
def plot_boxplot_subplot(ax, ec_data, eo_data, ylabel, label_letter):
    """
    This function plots a boxplot comparing EC vs EO data
    and adds statistical significance (t-test with asterisks).
    """
    # Perform independent t-test between Eyes Open (EO) and Eyes Closed (EC)
    t_stat, p_val = stats.ttest_ind(eo_data, ec_data, equal_var=False)

    # Convert p-values to asterisks for display
    def get_p_asterisks(p):
        if p < 0.0001:
            return "****"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # Define box colors: both in light blue for visual consistency
    colors = ['#ADD8E6', '#ADD8E6']

    # Draw the boxplot with black borders
    box = ax.boxplot([ec_data, eo_data], patch_artist=True,
                     boxprops=dict(color='black'),
                     medianprops=dict(color='black'))

    # Fill the boxes with the chosen colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set x-axis labels and style
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["EC", "EO"], fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)

    # Determine Y-axis limits with margins
    all_data = np.concatenate([ec_data, eo_data])
    y_max = np.max(all_data)
    y_min = np.min(all_data)
    margin_top = (y_max - y_min) * 0.4
    margin_bottom = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin_bottom, y_max + margin_top)

    # Draw statistical bar and asterisk annotation
    y_bar = y_max + margin_top * 0.5
    y_line = y_max + margin_top * 0.3
    ax.plot([1, 1, 2, 2], [y_line, y_bar, y_bar, y_line], lw=2, color='black')
    ax.text(1.5, y_bar + (margin_top * 0.05), get_p_asterisks(p_val),
            ha='center', va='bottom', color='black', fontsize=20, fontweight='bold')

    # Add subplot label (a), b), c))
    ax.text(-0.15, 1.08, label_letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------
# Load the EEG feature data (averaged by subject)
# These files contain feature values for the Alpha band (4-12 Hz)
# from Eyes Closed (EC) and Eyes Open (EO) conditions, WITHOUT artifact data
# ---------------------------------------------------------------

# Permutation Entropy (PE)
ec_without_pe = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_PE_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_pe = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_PE_ALPHA_4_1_W_59.npy"), axis=1)

# Skewness
ec_without_skew = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_skewness_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_skew = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_skewness_ALPHA_4_1_W_59.npy"), axis=1)

# Kurtosis
ec_without_kurt = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_kurtosis_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_kurt = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_kurtosis_ALPHA_4_1_W_59.npy"), axis=1)

# ---------------------------------------------------------------
# Create horizontal figure layout with 3 boxplots (PE, Skewness, Kurtosis)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# PE boxplot (a)
plot_boxplot_subplot(
    ax=axes[0],
    ec_data=ec_without_pe,
    eo_data=eo_without_pe,
    ylabel="PE",
    label_letra="a)"
)

# Skewness boxplot (b)
plot_boxplot_subplot(
    ax=axes[1],
    ec_data=ec_without_skew,
    eo_data=eo_without_skew,
    ylabel="Skewness",
    label_letra="b)"
)

# Kurtosis boxplot (c)
plot_boxplot_subplot(
    ax=axes[2],
    ec_data=ec_without_kurt,
    eo_data=eo_without_kurt,
    ylabel="Kurtosis",
    label_letra="c)"
)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("PE_Skewness_Kurtosis_alpha_WITHOUT.png", dpi=300)
plt.show()
