# Required libraries for statistics, data manipulation, and plotting
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Function to draw boxplots with statistical annotations for each subplot
def plot_boxplot_subplot(ax, ec_with, eo_with, ec_without, eo_without, ylabel, label_letter):
    # Data from both Eyes Closed (EC) and Eyes Open (EO) conditions
    data = [ec_with, eo_with, ec_without, eo_without]

    # Perform independent t-tests between EO and EC for both cases (with and without artifacts)
    t_stat_with, p_val_with = stats.ttest_ind(eo_with, ec_with, equal_var=False)
    t_stat_without, p_val_without = stats.ttest_ind(eo_without, ec_without, equal_var=False)

    # Define colors: pink for "with artifacts", light blue for "without artifacts"
    colors = ['#FFB6C1', '#FFB6C1', '#ADD8E6', '#ADD8E6']

    # Create the boxplots with black borders and markers
    box = ax.boxplot(data, patch_artist=True,
        boxprops=dict(color='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=6, linestyle='none')
    )

    # Fill boxes with the chosen colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set x-axis labels
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["EC", "EO", "EC", "EO"], fontsize=36, fontweight='bold')
    ax.tick_params(axis='both', labelsize=36)

    # Dynamically adjust Y-axis limits for spacing
    all_data = np.concatenate(data)
    y_max = np.max(all_data)
    y_min = np.min(all_data)
    margin_top = (y_max - y_min) * 0.5
    margin_bottom = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin_bottom, y_max + margin_top)

    # Helper function to convert p-values into asterisks for significance
    def get_p_asterisks(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"

    # Add p-value annotation for "with artifacts"
    y_bar_with = y_max + margin_top * 0.5
    y_line_with = y_max + margin_top * 0.3
    ax.plot([1, 1, 2, 2], [y_line_with, y_bar_with, y_bar_with, y_line_with], lw=2, color='black')
    ax.text(1.5, y_bar_with + (margin_top * 0.05), get_p_asterisks(p_val_with),
            ha='center', va='bottom', color='black', fontsize=36, fontweight='bold')

    # Add p-value annotation for "without artifacts"
    y_bar_without = y_max + margin_top * 0.8
    y_line_without = y_max + margin_top * 0.6
    ax.plot([3, 3, 4, 4], [y_line_without, y_bar_without, y_bar_without, y_line_without], lw=2, color='black')
    ax.text(3.5, y_bar_without + (margin_top * 0.05), get_p_asterisks(p_val_without),
            ha='center', va='bottom', color='black', fontsize=36, fontweight='bold')

    # Add subplot label (a), b), c)) and vertical metric label (⟨PE⟩, ⟨S⟩, ⟨K⟩)
    ax.text(-0.15, 1.05, label_letter, transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='top', ha='left')
    ax.text(-0.16, 0.5, rf"$\langle {ylabel} \rangle$", transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='center', ha='center', rotation=90)

# ----------------------------
# Load feature data (averaged per subject)
# ----------------------------

# Permutation Entropy (PE)
ec_with_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EC_pe_beta_4_1_w.npy"), axis=1)
eo_with_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EO_pe_beta_4_1_W.npy"), axis=1)
ec_without_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EC_pe_beta_4_1_wo.npy"), axis=1)
eo_without_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EO_pe_beta_4_1_Wo.npy"), axis=1)

# Skewness
ec_with_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EC_skew_beta_4_1_w.npy"), axis=1)
eo_with_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EO_skew_beta_4_1_W.npy"), axis=1)
ec_without_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EC_skew_beta_4_1_wo.npy"), axis=1)
eo_without_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EO_skew_beta_4_1_Wo.npy"), axis=1)

# Kurtosis
ec_with_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EC_kurt_beta_4_1_w.npy"), axis=1)
eo_with_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EO_kurt_beta_4_1_W.npy"), axis=1)
ec_without_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EC_kurt_beta_4_1_wo.npy"), axis=1)
eo_without_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EO_kurt_beta_4_1_Wo.npy"), axis=1)

# ----------------------------
# Plot vertical stacked boxplots (one per feature)
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(15, 30))  # 3 rows, 1 column layout

# Create each subplot for the 3 features
plot_boxplot_subplot(axes[0], ec_with_pe, eo_with_pe, ec_without_pe, eo_without_pe, "PE", "a)")
plot_boxplot_subplot(axes[1], ec_with_skew, eo_with_skew, ec_without_skew, eo_without_skew, "S", "b)")
plot_boxplot_subplot(axes[2], ec_with_kurt, eo_with_kurt, ec_without_kurt, eo_without_kurt, "K", "c)")

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(left=0.22)

# Save the final figure
plt.savefig("PE_Skewness_Kurtosis_VERTICAL_FINAL_BLACKEDGES.png", dpi=300)
plt.show()
