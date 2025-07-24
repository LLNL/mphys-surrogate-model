# compute the integrated diff between prediction intervals across latent space
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
from training_scripts.train_ae_ar import AEAutoregressor
from training_scripts.train_ae_sindy import AESINDy
from training_scripts.train_ae_NNdzdt import AENNdzdt

path = parent_directory + "/results/jonas-cp-tests/latent"

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
    alphas, DSD_bands_full, m_bands_full = pickle.load(f)

# load results from split conformal predictions
with open(path + "/split.pkl", "rb") as f:
    _, _, DSD_bands_split, m_bands_split = pickle.load(f)

model_label = ["SINDy", "NN-driven", "AR"]  # model labels

# compute differences between prediction bands
diff_full = DSD_bands_full[2] - DSD_bands_full[0]
diff_split = DSD_bands_split[2] - DSD_bands_split[0]
diff_full = diff_full[:, :, 0]  # these are the same across all samples
diff_split = diff_split[:, :, 0]

# compute difference between mass prediction bands
m_full = m_bands_full[2] - m_bands_full[0]
m_split = m_bands_split[2] - m_bands_split[0]
m_full = m_full[:, :, 0]  # these are the same across all samples
m_split = m_split[:, :, 0]
# average differences over latent space coordinates
avg_full = np.sum(diff_full, axis=-1) / diff_full.shape[-1]
avg_split = np.sum(diff_split, axis=-1) / diff_split.shape[-1]

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
(fig_m_full, ax_m_full) = plt.subplots(
    ncols=len(alphas),
    figsize=(2.5 * len(alphas), 2),
    sharey=True,
    constrained_layout=True,
)
(fig_m_split, ax_m_split) = plt.subplots(
    ncols=len(alphas),
    figsize=(2.5 * len(alphas), 2),
    sharey=True,
    constrained_layout=True,
)
for j, alpha in enumerate(alphas):  # loop through alpha values
    for i, m in enumerate(model_label):  # loop through models
        ax_full[j].plot(dsd_time, avg_full[i][j], label=m)
        ax_split[j].plot(dsd_time, avg_split[i][j], label=m)
        ax_m_full[j].plot(dsd_time, m_full[i][j], label=m)
        ax_m_split[j].plot(dsd_time, m_split[i][j], label=m)
    ax_full[j].set_title(r"$\alpha={}$".format(alpha))
    ax_split[j].set_title(r"$\alpha={}$".format(alpha))
    ax_m_full[j].set_title(r"$\alpha={}$".format(alpha))
    ax_m_split[j].set_title(r"$\alpha={}$".format(alpha))
    ax_full[j].set_xlabel("time [s]")
    ax_split[j].set_xlabel("time [s]")
    ax_m_full[j].set_xlabel("time [s]")
    ax_m_split[j].set_xlabel("time [s]")
    ax_full[j].set_ylim(-0.1 * avg_full.max(), 1.1 * avg_full.max())
    ax_split[j].set_ylim(-0.1 * avg_split.max(), 1.1 * avg_split.max())
    ax_m_full[j].set_ylim(-0.1 * m_full.max(), 1.1 * m_full.max())
    ax_m_split[j].set_ylim(-0.1 * m_split.max(), 1.1 * m_split.max())
ax_full[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_split[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_m_full[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_m_split[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig_full.suptitle(
    "Vanilla conformal DSD prediction interval width,\n averaged across latent space coordinates",
    fontsize=14,
)
fig_split.suptitle(
    "Split conformal DSD prediction interval width,\n averaged across latent space coordinates",
    fontsize=14,
)
fig_m_full.suptitle("Vanilla conformal mass prediction interval width", fontsize=14)
fig_m_split.suptitle("Split conformal mass prediction interval width", fontsize=14)
fig_full.savefig(
    parent_directory + "/results/jonas-cp-tests/latent/errors_vanilla_DSD.pdf"
)
fig_split.savefig(
    parent_directory + "/results/jonas-cp-tests/latent/errors_split_DSD.pdf"
)
fig_m_full.savefig(
    parent_directory + "/results/jonas-cp-tests/latent/errors_vanilla_m.pdf"
)
fig_m_split.savefig(
    parent_directory + "/results/jonas-cp-tests/latent/errors_split_m.pdf"
)
