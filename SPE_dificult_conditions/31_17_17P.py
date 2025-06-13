"""
This Python script generates a figure with two vertical boxplot panels to compare Spatial Permutation Entropy (SPE) values in Eyes Open (EO) and Eyes Closed (EC) conditions for two different EEG electrode montages (31 and 17 electrodes). It separates horizontal and vertical SPE patterns, performs statistical t-tests between EO and EC, and annotates the significance using asterisks.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# ======== CARGAR DATOS ========
hc_31 = np.load("vectores_31montaje/spe_hor_closed_raw_wo_31.npy")
ho_31 = np.load("vectores_31montaje/spe_hor_open_raw_wo_31.npy")
vc_31 = np.load("vectores_31montaje/spe_ver_closed_raw_wo_31.npy")
vo_31 = np.load("vectores_31montaje/spe_ver_open_raw_wo_31.npy")

hc_17 = np.load("vectores_17montaje/spe_hor_closed_raw_wo_17.npy")
ho_17 = np.load("vectores_17montaje/spe_hor_open_raw_wo_17.npy")
vc_17 = np.load("vectores_17montaje/spe_ver_closed_raw_wo_17.npy")
vo_17 = np.load("vectores_17montaje/spe_ver_open_raw_wo_17.npy")

# ======== PREPARAR DATOS ========
def split_and_stack(array, n_elect):
    return np.hstack(np.array_split(array, n_elect))

data_groups = [
    ("31 electrodes",
     [split_and_stack(hc_31, 31), split_and_stack(ho_31, 31)],
     [split_and_stack(vc_31, 31), split_and_stack(vo_31, 31)]),

    ("17 electrodes",
     [split_and_stack(hc_17, 17), split_and_stack(ho_17, 17)],
     [split_and_stack(vc_17, 17), split_and_stack(vo_17, 17)])
]

# ======== FIGURA Y SUBPLOTS ========
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
letters = ['a)', 'b)']

for i, (_, hor_data, ver_data) in enumerate(data_groups):
    ax = axs[i]
    all_data = hor_data + ver_data
    positions = [1, 2, 4, 5]
    labels = ["EC", "EO", "EC", "EO"]

    # Boxplots estrechos y claros
    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.4,
                    medianprops=dict(color='black'))
    for patch in bp['boxes']:
        patch.set_facecolor("#ADD8E6")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18, fontweight='bold')  # ← aumentado
    ax.tick_params(axis='y', labelsize=18)  

    y_max = max([np.max(d) for d in all_data])
    y_min = min([np.min(d) for d in all_data])
    margen = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - margen, y_max + margen * 1.5)

    for j, (a, b) in enumerate([(1, 2), (4, 5)]):
        t_stat, p_val = stats.ttest_ind(all_data[j*2], all_data[j*2+1], equal_var=False)
        y_line = y_max + margen * 0.1 + j * margen * 0.2
        y_bar = y_line + margen * 0.05
        ax.plot([a, a, b, b], [y_line, y_bar, y_bar, y_line], lw=2, color='black')
        ax.text((a + b) / 2, y_bar + margen*0.02, get_p_asterisks(p_val),
                ha='center', va='bottom', fontsize=16, fontweight='bold')  

    ax.set_ylabel("SPE", fontsize=20, fontweight='bold')  
    ax.text(0.01, 1.05, letters[i], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left') 

    ax.text(0.28, -0.12, "Horizontal", ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes) 
    ax.text(0.72, -0.12, "Vertical", ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes) 

# ======== AJUSTAR Y GUARDAR ========
fig.align_ylabels(axs)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig('SPE_31_17_vertical_alineado.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()
