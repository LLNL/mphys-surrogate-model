# compute the integrated area between prediction intervals across the entire network output
# these integrals will naturally be a function of time, weighted by the domain size

import os
import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)

from src import data_utils as du

path = parent_directory + "/results/jonas-cp-tests/full"

(
    x_train,
    m_train,
    x_test,
    m_test,
    r_bins_edges,
    n_bins,
    dsd_time,
) = du.open_erf_dataset(
    sample_time=np.arange(0, 61, 5),
    path=parent_directory + "/data/congestus_coal_200m",
)

# load results from vanilla conformal predictions
with open(path + "/vanilla.pkl", "rb") as f:
    alphas, DSD_bands_full, _ = pickle.load(f)

# load results from split conformal predictions
with open(path + "/split.pkl", "rb") as f:
    _, _, DSD_bands_split, _ = pickle.load(f)

model_label = ["SINDy", "NN-driven", "AR"]  # model labels

# compute differences between prediction bands
diff_full = DSD_bands_full[2] - DSD_bands_full[0]
diff_split = DSD_bands_split[2] - DSD_bands_split[0]
diff_full = diff_full[:, :, 0]  # these are the same across all samples
diff_split = diff_split[:, :, 0]

# compute areas between prediction bands
# compute midpoints of bins
rbins_mid = (
    np.log(r_bins_edges.rbin_l.values) + np.log(r_bins_edges.rbin_r.values)
) / 2
# composite trapezoidal rule
area_full = np.trapz(diff_full, x=rbins_mid, axis=-1)
area_split = np.trapz(diff_split, x=rbins_mid, axis=-1)

# define figures. Columns correspond to alpha values
(fig_full, ax_full) = plt.subplots(
    ncols=len(alphas),
    figsize=(2.5 * len(alphas), 2),
    sharey=True,
    constrained_layout=True,
)
(fig_split, ax_split) = plt.subplots(
    ncols=len(alphas),
    figsize=(2.5 * len(alphas), 2),
    sharey=True,
    constrained_layout=True,
)
for j, alpha in enumerate(alphas):  # loop through alpha values
    for i, m in enumerate(model_label):  # loop through models
        ax_full[j].plot(dsd_time, area_full[i][j], label=m)
        ax_split[j].plot(dsd_time, area_split[i][j], label=m)
    ax_full[j].set_title(r"$\alpha={}$".format(alpha))
    ax_split[j].set_title(r"$\alpha={}$".format(alpha))
    ax_full[j].set_xlabel("time [s]")
    ax_split[j].set_xlabel("time [s]")
    ax_full[j].set_ylim(0.9 * area_full.min(), 1.1 * area_full.max())
    ax_split[j].set_ylim(0.9 * area_split.min(), 1.1 * area_split.max())
ax_full[0].set_ylabel("mass mixing ratio \n [kg liquid/kg air]")
ax_split[0].set_ylabel("mass mixing ratio \n [kg liquid/kg air]")
ax_full[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_split[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig_full.suptitle(
    "Vanilla conformal DSD prediction interval width,\n integrated across DSD bins",
    fontsize=14,
)
fig_split.suptitle(
    "Split conformal DSD prediction interval width,\n integrated across DSD bins",
    fontsize=14,
)
fig_full.savefig(parent_directory + "/results/jonas-cp-tests/full/errors_vanilla.pdf")
fig_split.savefig(parent_directory + "/results/jonas-cp-tests/full/errors_split.pdf")
