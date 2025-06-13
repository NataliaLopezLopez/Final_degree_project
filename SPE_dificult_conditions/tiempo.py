import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# ---------- Cargar datos ----------
hor_closed = np.load('vectores/spe_hor_closed_raw_wo.npy')[:, :1000]
hor_open = np.load('vectores/spe_hor_open_raw_wo.npy')[:, :1000]
ver_closed = np.load('vectores/spe_ver_closed_raw_wo.npy')[:, :1000]
ver_open = np.load("vectores/spe_ver_open_raw_wo.npy")[:, :1000]

time_points = np.arange(1, hor_open.shape[1] + 1)

# ---------- Funciones auxiliares ----------
def compute_mean_over_time(data):
    means, stds = [], []
    for t in time_points:
        subject_means = np.mean(data[:, :t], axis=1)
        means.append(np.mean(subject_means))
        stds.append(np.std(subject_means))
    return np.array(means), np.array(stds)

def compute_temporal_pvalues(open_data, closed_data):
    pvalues = []
    for t in time_points:
        data_open = np.mean(open_data[:, :t], axis=1)
        data_closed = np.mean(closed_data[:, :t], axis=1)
        pval = ttest_rel(data_open, data_closed).pvalue
        pvalues.append(pval)
    return np.array(pvalues)

# ---------- Calcular métricas ----------
mean_hor_open, std_hor_open = compute_mean_over_time(hor_open)
mean_hor_closed, std_hor_closed = compute_mean_over_time(hor_closed)
pval_hor = compute_temporal_pvalues(hor_open, hor_closed)

mean_ver_open, std_ver_open = compute_mean_over_time(ver_open)
mean_ver_closed, std_ver_closed = compute_mean_over_time(ver_closed)
pval_ver = compute_temporal_pvalues(ver_open, ver_closed)

# ---------- Crear figura ----------
fig, axs = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)

# ---------- Subplot a): Horizontal ----------
ax1 = axs[0]
lns1_1 = ax1.plot(time_points, mean_hor_open, label='Eyes Open', color='blue')
ax1.fill_between(time_points, mean_hor_open - std_hor_open, mean_hor_open + std_hor_open, alpha=0.3, color='blue')
lns1_2 = ax1.plot(time_points, mean_hor_closed, label='Eyes Closed', color='magenta')
ax1.fill_between(time_points, mean_hor_closed - std_hor_closed, mean_hor_closed + std_hor_closed, alpha=0.3, color='magenta')
ax1.set_ylabel("Average SPE", fontsize=19)
ax1.tick_params(labelsize=19)
ax1.grid()

ax1r = ax1.twinx()
lns1_3 = ax1r.plot(time_points, pval_hor, color='black', label='p-value')
ax1r.set_yscale('log')
ax1r.set_ylabel("p-value", fontsize=19)
ax1r.tick_params(labelsize=19)

# ---------- Subplot b): Vertical ----------
ax2 = axs[1]
lns2_1 = ax2.plot(time_points, mean_ver_open, label='Eyes Open', color='blue')
ax2.fill_between(time_points, mean_ver_open - std_ver_open, mean_ver_open + std_ver_open, alpha=0.3, color='blue')
lns2_2 = ax2.plot(time_points, mean_ver_closed, label='Eyes Closed', color='magenta')
ax2.fill_between(time_points, mean_ver_closed - std_ver_closed, mean_ver_closed + std_ver_closed, alpha=0.3, color='magenta')
ax2.set_xlabel("Analyzed time (sampling points)", fontsize=19)
ax2.set_ylabel("Average SPE", fontsize=19)
ax2.tick_params(labelsize=19)
ax2.grid()

ax2r = ax2.twinx()
lns2_3 = ax2r.plot(time_points, pval_ver, color='black', label='p-value')
ax2r.set_yscale('log')
ax2r.set_ylabel("p-value", fontsize=19)
ax2r.tick_params(labelsize=19)


fig.text(-0.02, 0.91, 'a)', fontsize=19, fontweight='bold')
fig.text(-0.02, 0.43, 'b)', fontsize=19, fontweight='bold')


lns = lns1_1 + lns1_2 + lns1_3
labels = [l.get_label() for l in lns]
fig.legend(lns, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=18, frameon=False)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figura6_SPE_64.png", dpi=300, bbox_inches='tight')
plt.show()
