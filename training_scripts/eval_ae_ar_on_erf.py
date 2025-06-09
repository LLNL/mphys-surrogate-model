import sys
import os
import time
import copy
import pysindy as ps

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import xarray as xr
import torch
import pickle as pkl
import uuid
from src import data_utils as du, models, training, plotting
from torch.utils.data import DataLoader
from train_ae_ar import AEAutoregressor, train_model

load_params = {
    "data_src": "box",
    "random_seed": 10,
    "num_epochs": 30,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "n_lag": 1,
    "w_recon": 1,
    "w_dx": 1,
    "w_dz": 1,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "layer_size": (20, 20, 10),
    "CNN": False,
    "print_frequency": 1,
    "id": "f1b52d06fe99407ba234710eb604a61d",
}
output_directory = "../trained_models/ae_ar_normed"
test_ids = [0, 10, 20, 30]
tplt = [0, 30, -1]
# tplt = [0, 2, 5, 8, -1]
if load_params["CNN"]:
    prefix = "CNN"
else:
    prefix = "FFNN"

case_name = prefix + "_latent{}_order{}_tr{}_lr{}_bs{}_weights{}-{}_{}".format(
    load_params["latent_dim"],
    load_params["layer_size"],
    load_params["num_epochs"],
    load_params["learning_rate"],
    load_params["batch_size"],
    load_params["w_dx"],
    load_params["w_dz"],
    load_params["id"],
)
print("Loading " + output_directory + "/" + case_name)


params = load_params.copy()
params["data_src"] = "erf"
# params["lr_sched"] = False
params["batch_size"] = 128
params["wd"] = 0.0
params["learning_rate"] = 1e-2
# params["w_dx"] = 0
# params["w_recon"] = 0
params["num_epochs"] = 50
if params["data_src"] == "box":
    (
        x_train,
        m_train,
        x_test,
        m_test,
        r_bins_edges,
        n_bins,
        dsd_time,
    ) = du.open_box_dataset()
elif params["data_src"] == "erf":
    (
        x_train,
        m_train,
        x_test,
        m_test,
        r_bins_edges,
        n_bins,
        dsd_time,
    ) = du.open_erf_dataset()
    x_test = x_test[:, :, :-1]
    x_train = x_train[:, :, :-1]
    r_bins_edges = r_bins_edges[:-1]
    n_bins = n_bins - 1

train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=params["batch_size"], shuffle=True
)
test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=len(test_data), shuffle=True
)

#
model = AEAutoregressor(
    n_channels=1,
    n_bins=n_bins,
    n_latent=params["latent_dim"],
    n_lag=params["n_lag"],
    CNN=params["CNN"],
)
model.load_state_dict(torch.load(output_directory + "/model/" + case_name + ".pth"))

# plotting.plot_reconstructions(
#         model,
#         test_ids,
#         x_test,
#         r_bins_edges,
#         #saveas=output_directory + "/plots/" + case_name + "_reconstructions_erf.png",
#     )
#
# plotting.plot_predictions_AE_AR(
#         model,
#         test_ids,
#         dsd_time,
#         tplt,
#         x_test,
#         m_test,
#         r_bins_edges,
#         #saveas=output_directory + "/plots/" + case_name + "_predictions_erf.png",
#     )
#
# plotting.plot_latent_trajectories_AR(
#         params["latent_dim"],
#         model,
#         dsd_time,
#         x_test,
#         m_test,
#         # saveas=output_directory + "/plots/" + case_name + "_trajectories.png",
#     )
#
# plotting.viz_3d_latent_space(
#     model,
#     x_test,
#     dsd_time,
#     output_directory + "/plots/",
#     case_name + "_erfeval",
# )

# refit AR portion of model
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.AdamW(
#     model.autoregressor.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
# )
optimizer = torch.optim.AdamW(
    model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
early_stopping = training.EarlyStopping(patience=params["patience"])

total_params = sum(p.numel() for p in model.parameters())
ar_params = sum(p.numel() for p in model.autoregressor.parameters())
print(f"Retraining {ar_params} of {total_params} parameters")

# model.autoregressor = models.Autoregressive(
#             n_bins=params["latent_dim"] + 1,
#             n_bins_in=params["latent_dim"] * params["n_lag"] + 1,
#             layer_size=params["layer_size"],
#         )

best_model, (
    losses,
    recon_losses,
    dx_losses,
    dz_losses,
    test_losses,
    test_recon_losses,
    test_dx_losses,
    test_dz_losses,
) = train_model(
    params,
    model,
    train_loader,
    test_loader,
    divergence,
    criterion,
    optimizer,
    early_stopping,
    sched,
)

plotting.plot_losses(
    losses,
    test_losses=test_losses,
    sub_losses=[
        params["w_dx"] * dx_losses,
        params["w_dz"] * dz_losses,
        params["w_recon"] * recon_losses,
    ],
    labels=["X: t -> t+1", "Z: t -> t+1", "Recon"],
    title=f"Training Loss, lag {params['n_lag']}",
    # saveas=output_directory + "/plots/" + case_name + "_losses.png",
)

plotting.plot_reconstructions(
    model,
    test_ids,
    x_test,
    r_bins_edges,
    # saveas=output_directory + "/plots/" + case_name + "_reconstructions_erf.png",
)

plotting.plot_predictions_AE_AR(
    best_model,
    test_ids,
    dsd_time,
    tplt,
    x_test,
    m_test,
    r_bins_edges,
    # saveas=output_directory + "/plots/" + case_name + "_predictions.png",
)
# Plot trajectories of the latent variables
plotting.plot_latent_trajectories_AR(
    params["latent_dim"],
    best_model,
    dsd_time,
    x_test,
    m_test,
    # saveas=output_directory + "/plots/" + case_name + "_trajectories.png",
)

plotting.viz_3d_latent_space(
    model,
    x_test,
    dsd_time,
    output_directory + "/plots/",
    case_name + "_erf-retrain",
)
