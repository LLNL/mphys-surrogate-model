import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    sample_time=np.arange(0, 61, 5), path=parent_directory + "/data/congestus_coal_200m"
)

# load results from vanilla conformal predictions
with open(path + "/vanilla.pkl", "rb") as f:
    alphas, DSD_bands_full, _ = pickle.load(f)

# load results from split conformal predictions
with open(path + "/split.pkl", "rb") as f:
    _, idx_testing, DSD_bands_split, _ = pickle.load(f)

test_ids_raw = [
    0,
    10,
    20,
    30,
]  # indices for trajectories to plot, indexed according to split conformal
test_ids = idx_testing[test_ids_raw]  # trajectories to plot
tplt = [0, 5, 11]  # which times to plot

model_label = ["SINDy", "NN-driven", "AR"]  # model labels

lower_full = DSD_bands_full[0]
rep_full = DSD_bands_full[1]
upper_full = DSD_bands_full[2]

lower_split = DSD_bands_split[0]
rep_split = DSD_bands_split[1]
upper_split = DSD_bands_split[2]

# use alphas to get color arguments for fill_between (the prediction intervals on the plot)
cmap = plt.get_cmap("autumn_r")  # 0→yellow, 1→red
colors = cmap(
    np.tanh(np.pi * np.array(alphas))
)  # take tanh to ensure that it is mostly red until very close to 0

for k, model in enumerate(model_label):
    (fig_full, ax_full) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(2.5 * len(test_ids), 2 * len(tplt)),
        sharey=True,
    )
    (fig_split, ax_split) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(2.5 * len(test_ids), 2 * len(tplt)),
        sharey=True,
    )
    for i, id in enumerate(test_ids_raw):
        for j, t in enumerate(tplt):
            ax_full[j][i].step(r_bins_edges, x_test[test_ids[i], t], label="Data")
            ax_split[j][i].step(r_bins_edges, x_test[test_ids[i], t], label="Data")
            for k_alpha, alpha in enumerate(alphas):
                ax_full[j][i].fill_between(
                    r_bins_edges,
                    np.maximum(0, lower_full[k, k_alpha, test_ids[i], t]),
                    upper_full[k, k_alpha, test_ids[i], t],
                    step="pre",
                    color=colors[k_alpha],
                    alpha=0.3,
                    label=f"{100*(1-alpha)}% coverage",
                )
                ax_split[j][i].fill_between(
                    r_bins_edges,
                    np.maximum(0, lower_split[k, k_alpha, id, t]),
                    upper_split[k, k_alpha, id, t],
                    step="pre",
                    color=colors[k_alpha],
                    alpha=0.3,
                    label=f"{100*(1-alpha)}% coverage",
                )
            ax_full[j][i].step(
                r_bins_edges, rep_full[k, test_ids[i], t], label=model
            )  # representative band
            ax_split[j][i].step(
                r_bins_edges, rep_split[k, id, t], label=model
            )  # representative band
            ax_full[j][i].set_xscale("log")
            ax_split[j][i].set_xscale("log")
        ax_full[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_full[-1][i].set_xlabel("radius [µm]")
        ax_split[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_split[-1][i].set_xlabel("radius [µm]")
        ax_full[j][i].set_ylim(0, 0.5)
        ax_split[j][i].set_ylim(0, 0.5)
    for j, t in enumerate(tplt):
        ax_full[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
        ax_split[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax_full[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_split[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig_full.suptitle("Vanilla conformal DSD predictions", fontsize=14)
    fig_split.suptitle("Split conformal DSD predictions", fontsize=14)
    plt.tight_layout()
    fig_full.savefig(
        f"{parent_directory}/results/jonas-cp-tests/full/compare_vanilla_{model}.pdf",
        bbox_inches="tight",
    )
    fig_split.savefig(
        f"{parent_directory}/results/jonas-cp-tests/full/compare_split_{model}.pdf",
        bbox_inches="tight",
    )
    fig_full.clf()
    fig_split.clf()
