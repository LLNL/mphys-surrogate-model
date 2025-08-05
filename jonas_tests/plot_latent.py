import os
import sys
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)

from src import data_utils as du
from training_scripts.train_ae_ar import AEAutoregressor
from training_scripts.train_ae_sindy import AESINDy
from training_scripts.train_ae_NNdzdt import AENNdzdt

path = parent_directory + "/results/jonas_cp_tests/latent"

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

params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "n_lag": 1,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    # "lambda3_sparsity": 0.0, TODO: sequential thresholding
    "CNN": False,
    "print_frequency": 1,
}

main_path = parent_directory + "/results/poster_erf_results/"
ae_ar_checkpoint = torch.load(
    main_path
    + "ae_ar/model/erf_FFNN_latent3_order(10, 20, 10)_tr100_lr0.001_bs128_weights1-1_f74ecd55b87a43e2bfde2bf9973fdf10.pth",
    weights_only=True,
)
ae_nndzdt_checkpoint = torch.load(
    main_path
    + "ae_nndzdt/model/FFNN_latent3_layers(40, 40, 40)_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_7850dd2260f34875baf55452f4311d60.pth",
    weights_only=True,
)
ae_sindy_checkpoint = torch.load(
    main_path
    + "ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth",
    weights_only=True,
)

# %%
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(ae_sindy_checkpoint)
# %%
ae_nndzdt = AENNdzdt(
    n_channels=1, n_bins=64, n_latent=3, layer_size=(40, 40, 40), CNN=False
)
ae_nndzdt.load_state_dict(ae_nndzdt_checkpoint)
# %%
ae_ar = AEAutoregressor(
    n_channels=1, n_bins=64, n_latent=3, n_lag=1, layer_size=(10, 20, 10), CNN=False
)
ae_ar.load_state_dict(ae_ar_checkpoint)

# load results from vanilla conformal predictions
with open(path + "/vanilla.pkl", "rb") as f:
    alphas, DSD_bands_full, m_bands_full = pickle.load(f)

# load results from split conformal predictions
with open(path + "/split.pkl", "rb") as f:
    _, idx_testing, DSD_bands_split, m_bands_split = pickle.load(f)

test_ids_raw = [
    0,
    10,
    20,
    30,
]  # indices for trajectories to plot, indexed according to split conformal
test_ids = idx_testing[test_ids_raw]  # trajectories to plot

models = (ae_sindy, ae_nndzdt, ae_ar)
model_label = ["SINDy", "NN-driven", "AR"]  # model labels

lower_full = DSD_bands_full[0]
rep_full = DSD_bands_full[1]
upper_full = DSD_bands_full[2]

m_lower_full = m_bands_full[0]
m_rep_full = m_bands_full[1]
m_upper_full = m_bands_full[2]

lower_split = DSD_bands_split[0]
rep_split = DSD_bands_split[1]
upper_split = DSD_bands_split[2]

m_lower_split = m_bands_split[0]
m_rep_split = m_bands_split[1]
m_upper_split = m_bands_split[2]

# use alphas to get color arguments for fill_between (the prediction intervals on the plot)
cmap = plt.get_cmap("autumn_r")  # 0→yellow, 1→red
colors = cmap(
    np.tanh(np.pi * np.array(alphas))
)  # take tanh to ensure that it is mostly red until very close to 0

(fig_full_m, ax_full_m) = plt.subplots(
    ncols=len(test_ids),
    nrows=len(model_label),
    figsize=(2.5 * len(test_ids), 2 * 3),
    sharey=True,
)  # masses. Rows correspond to models.
(fig_split_m, ax_split_m) = plt.subplots(
    ncols=len(test_ids),
    nrows=len(model_label),
    figsize=(2.5 * len(test_ids), 2 * 3),
    sharey=True,
)  # masses. Rows correspond to models.

for k, model in enumerate(model_label):
    z_enc_test = models[k].encoder(torch.Tensor(x_test)).detach().numpy()
    # columns correspond to test_ids, rows correpsond to latent dimension
    (fig_full, ax_full) = plt.subplots(
        ncols=len(test_ids),
        nrows=3,
        figsize=(2.5 * len(test_ids), 2 * 3),
        sharey=True,
    )  # vanilla/full conformal. Rows correspond to latent variables.
    (fig_split, ax_split) = plt.subplots(
        ncols=len(test_ids),
        nrows=3,
        figsize=(2.5 * len(test_ids), 2 * 3),
        sharey=True,
    )  # split conformal. Rows correspond to latent variables.
    for i, id in enumerate(test_ids_raw):
        ax_full_m[k][i].plot(dsd_time, m_test[test_ids[i]], label="Data")
        ax_split_m[k][i].plot(dsd_time, m_test[test_ids[i]], label="Data")
        for k_alpha, alpha in enumerate(
            alphas
        ):  # ensure masses are always non-negative
            ax_full_m[k][i].fill_between(
                dsd_time,
                np.maximum(0, m_lower_full[k, k_alpha, test_ids[i]]),
                m_upper_full[k, k_alpha, test_ids[i]],
                color=colors[k_alpha],
                alpha=0.3,
                label=f"{100*(1-alpha)}% coverage",
            )
            ax_split_m[k][i].fill_between(
                dsd_time,
                np.maximum(0, m_lower_split[k, k_alpha, id]),
                m_upper_split[k, k_alpha, id],
                color=colors[k_alpha],
                alpha=0.3,
                label=f"{100*(1-alpha)}% coverage",
            )
        ax_full_m[k][i].plot(
            dsd_time, m_rep_full[k, test_ids[i]], label="Model"
        )  # representative band
        ax_split_m[k][i].plot(
            dsd_time, m_rep_split[k, id], label="Model"
        )  # representative band
        for j in range(3):
            ax_full[j][i].plot(dsd_time, z_enc_test[test_ids[i], :, j], label="Data")
            ax_split[j][i].plot(dsd_time, z_enc_test[test_ids[i], :, j], label="Data")
            for k_alpha, alpha in enumerate(alphas):
                ax_full[j][i].fill_between(
                    dsd_time,
                    lower_full[k, k_alpha, test_ids[i], :, j],
                    upper_full[k, k_alpha, test_ids[i], :, j],
                    color=colors[k_alpha],
                    alpha=0.3,
                    label=f"{100*(1-alpha)}% coverage",
                )
                ax_split[j][i].fill_between(
                    dsd_time,
                    lower_split[k, k_alpha, id, :, j],
                    upper_split[k, k_alpha, id, :, j],
                    color=colors[k_alpha],
                    alpha=0.3,
                    label=f"{100*(1-alpha)}% coverage",
                )
            ax_full[j][i].plot(
                dsd_time, rep_full[k, test_ids[i], :, j], label=model
            )  # representative band
            ax_split[j][i].plot(
                dsd_time, rep_split[k, id, :, j], label=model
            )  # representative band
        ax_full[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_full[-1][i].set_xlabel("time [s]")
        ax_split[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_split[-1][i].set_xlabel("time [s]")
        ax_full_m[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_full_m[-1][i].set_xlabel("time [s]")
        ax_split_m[0][i].set_title(f"Sample #{test_ids[i]}")
        ax_split_m[-1][i].set_xlabel("time [s]")
    ax_full_m[k][0].set_ylabel(model + "\n normalized mass [-]")
    ax_split_m[k][0].set_ylabel(model + "\n normalized mass [-]")
    for j in range(3):
        ax_full[j][0].set_ylabel(r"$z_{}$ [-]".format(j + 1))
        ax_split[j][0].set_ylabel(r"$z_{}$ [-]".format(j + 1))
    ax_full[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_split[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig_full.suptitle("Vanilla conformal latent space predictions", fontsize=14)
    fig_split.suptitle("Split conformal latent space predictions", fontsize=14)
    plt.tight_layout()
    fig_full.savefig(
        f"{parent_directory}/results/jonas_cp_tests/latent/compare_vanilla_{model}.pdf",
        bbox_inches="tight",
    )
    fig_split.savefig(
        f"{parent_directory}/results/jonas_cp_tests/latent/compare_split_{model}.pdf",
        bbox_inches="tight",
    )
    fig_full.clf()
    fig_split.clf()
ax_full_m[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax_split_m[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig_full_m.suptitle("Vanilla conformal latent space predictions", fontsize=14)
fig_split_m.suptitle("Split conformal latent space predictions", fontsize=14)
plt.tight_layout()
fig_full_m.savefig(
    f"{parent_directory}/results/jonas_cp_tests/latent/compare_vanilla_m.pdf",
    bbox_inches="tight",
)
fig_split_m.savefig(
    f"{parent_directory}/results/jonas_cp_tests/latent/compare_split_m.pdf",
    bbox_inches="tight",
)
