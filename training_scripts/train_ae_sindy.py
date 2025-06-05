import sys
import os
import time
import copy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import xarray as xr
import torch
import pickle as pkl
import uuid
from src import data_utils as du, models, training, plotting
from torch.utils.data import DataLoader

params = {
    "random_seed": 10,
    "num_epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    # "lambda3_sparsity": 0.0, TODO: sequential thresholding
    "CNN": False,
    "print_frequency": 1,
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])


class AESINDy(torch.nn.Module):
    def __init__(self, n_channels=1, n_bins=100, n_latent=10, poly_order=2, CNN=False):
        super(AESINDy, self).__init__()
        self.poly_order = poly_order

        if CNN:
            self.encoder = models.CNNEncoder(
                n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
            )
            self.decoder = models.CNNDecoder(
                n_channels=n_channels,
                n_bins=n_bins,
                n_latent=n_latent,
                distribution=True,
            )

        else:
            assert n_channels == 1
            self.encoder = models.FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
            self.decoder = models.FFNNDecoder(
                n_bins=n_bins, n_latent=n_latent, distribution=True
            )
        self.dzdt = models.SINDyDeriv(n_latent=n_latent + 1, poly_order=poly_order)

    def forward(self, bin0, M):
        z0 = self.encoder(bin0)
        dzMdt = self.dzdt(z0, M)
        dzdt = dzMdt[:, :, :-1]
        return dzdt


if __name__ == "__main__":
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # torch.backends.cudnn.benchmark = True
    print(f"Using {device} device")

    start_time = time.time()
    # Open dataset
    ds_all = xr.open_dataset("../data/box64_train.nc")
    dlnr = np.diff(np.log(ds_all["mass_bin"].values)).mean()
    m_train = ds_all["dvdlnr"].sum(dim="mass_bin_idx")
    x_train = (
        (ds_all["dvdlnr"] / m_train).transpose("run", "time", "mass_bin_idx").to_numpy()
    )
    m_scale = m_train.max()
    m_train = (m_train / m_scale).to_numpy()
    n_bins = x_train.shape[2]
    dsd_time = (ds_all["time"] / np.timedelta64(1, "s")).to_numpy()

    ds_test = xr.open_dataset("../data/box64_test.nc")
    m_test = ds_test["dvdlnr"].sum(dim="mass_bin_idx")
    x_test = (
        (ds_test["dvdlnr"] / m_test).transpose("run", "time", "mass_bin_idx").to_numpy()
    )
    m_test = (m_test / m_scale).to_numpy()

    train_data = du.NormedBinDatasetDzDt(x_train, dsd_time, m_train)
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    test_data = du.NormedBinDatasetDzDt(x_test, dsd_time, m_test)
    test_loader = DataLoader(test_data, batch_size=x_test.shape[0], shuffle=True)

    # Initialize the model
    model = AESINDy(
        n_channels=1,
        n_bins=n_bins,
        n_latent=params["latent_dim"],
        poly_order=params["poly_order"],
        CNN=params["CNN"],
    )

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stopping = training.EarlyStopping(patience=params["patience"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Compute & set weights based on Champion et al recs
    xx = np.squeeze(train_data.x)
    dx = np.squeeze(train_data.dx)
    xxl2 = np.linalg.norm(xx, ord=2, axis=1) ** 2
    dxl2 = np.linalg.norm(dx, ord=2, axis=1) ** 2
    lambda1 = xxl2.sum() / dxl2.sum() * params["lambda1_factor"]
    lambda2 = lambda1 / 1e2  # 2 orders of magnitude smaller
    print(f"lambda: 1.0, {lambda1}, {lambda2}")
    params["loss_weight_recon"] = 1.0
    params["loss_weight_sindy_x"] = lambda1
    params["loss_weight_sindy_z"] = lambda2

    # Set up loss storage
    losses = np.zeros(params["num_epochs"]) * np.nan
    recon_losses = np.zeros(params["num_epochs"]) * np.nan
    dx_losses = np.zeros(params["num_epochs"]) * np.nan
    dz_losses = np.zeros(params["num_epochs"]) * np.nan

    test_losses = np.zeros(params["num_epochs"]) * np.nan
    test_recon_losses = np.zeros(params["num_epochs"]) * np.nan
    test_dx_losses = np.zeros(params["num_epochs"]) * np.nan
    test_dz_losses = np.zeros(params["num_epochs"]) * np.nan
    best_test_loss = float("inf")

    for epoch in range(params["num_epochs"]):
        # Train
        epoch_start_time = time.time()
        model.train()
        for batch_x, batch_dx, batch_M in train_loader:
            # Forward pass
            pred_x_recon = model.decoder(model.encoder(batch_x))
            z = model.encoder(batch_x)
            zz = z.clone().detach().requires_grad_()
            pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
            _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
            _, pred_dx = torch.func.jvp(model.decoder, (zz,), (pred_dz,))

            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_x + params["tol"]),
            )
            loss_dx = criterion(pred_dx, batch_dx)
            loss_dz = criterion(pred_dz, dz)

            loss = (
                params["loss_weight_sindy_x"] * loss_dx
                + params["loss_weight_recon"] * loss_recon
                + params["loss_weight_sindy_z"] * loss_dz
            )

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Save train losses
        losses[epoch] = loss.item()
        recon_losses[epoch] = loss_recon.item()
        dx_losses[epoch] = loss_dx.item()
        dz_losses[epoch] = loss_dz.item()

        # test
        model.eval()
        for batch_x, batch_dx, batch_M in test_loader:
            # Forward pass
            pred_x_recon = model.decoder(model.encoder(batch_x))
            z = model.encoder(batch_x)
            zz = z.clone().detach().requires_grad_()
            pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
            _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
            _, pred_dx = torch.func.jvp(model.decoder, (zz,), (pred_dz,))

            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_x + params["tol"]),
            )
            loss_dx = criterion(pred_dx, batch_dx)
            loss_dz = criterion(pred_dz, dz)

            loss = (
                params["loss_weight_sindy_x"] * loss_dx
                + params["loss_weight_recon"] * loss_recon
                + params["loss_weight_sindy_z"] * loss_dz
            )

        # Save test losses
        test_losses[epoch] = loss.item()
        test_recon_losses[epoch] = loss_recon.item()
        test_dx_losses[epoch] = loss_dx.item()
        test_dz_losses[epoch] = loss_dz.item()

        # Save good model
        if loss < best_test_loss:
            best_test_loss = loss
            best_model = copy.deepcopy(model)

        # Update learning rate schedule
        if params["lr_sched"]:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(loss)
            else:
                sched.step()

        # print progress
        epoch_end_time = time.time()
        if epoch % params["print_frequency"] == 0:
            print(
                f"Epoch [{epoch}/{params['num_epochs']}], Train Loss: {losses[epoch]:.4f} | "
                f"Test Loss: {test_losses[epoch]:.4f} | LR: {sched.get_last_lr()}"
                f"| Epoch Time: {epoch_end_time - epoch_start_time} s"
            )
            print(
                f"Recon: {params['loss_weight_recon'] * recon_losses[epoch]:.4f} | "
                f"dx: {params['loss_weight_sindy_x'] * dx_losses[epoch]:.4f} | "
                f"dz: {params['loss_weight_sindy_z'] * dz_losses[epoch]:.4f} | "
            )

        # early stopping
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    # SAVE
    best_model.eval()
    output_directory = "../trained_models/ae_sindy_normed"
    id = uuid.uuid4().hex
    if params["CNN"]:
        prefix = "CNN"
    else:
        prefix = "FFNN"
    case_name = prefix + "_latent{}_order{}_tr{}_lr{}_bs{}_weights{}-{}-{}_{}".format(
        params["latent_dim"],
        params["poly_order"],
        params["num_epochs"],
        params["learning_rate"],
        params["batch_size"],
        params["loss_weight_recon"],
        params["loss_weight_sindy_z"],
        params["loss_weight_sindy_x"],
        id,
    )
    with open(output_directory + "/losses/" + case_name + ".pkl", "wb") as pickle_file:
        pkl.dump(
            (
                losses,
                recon_losses,
                dx_losses,
                dz_losses,
                test_losses,
                test_recon_losses,
                test_dx_losses,
                test_dz_losses,
            ),
            pickle_file,
        )
    torch.save(
        best_model.state_dict(), output_directory + "/model/" + case_name + ".pth"
    )
    print(f"Saved model and losses as {case_name}")

    ############### PLOTS PLOTS PLOTS ###############
    # Plot training loss
    plotting.plot_losses(
        losses,
        test_losses=test_losses,
        sub_losses=[
            params["loss_weight_sindy_x"] * np.array(dx_losses),
            params["loss_weight_sindy_z"] * np.array(dz_losses),
            params["loss_weight_recon"] * np.array(recon_losses),
        ],
        labels=["dx/dt", "dz/dt", "Recon"],
        title=f"Training Loss",
        saveas=output_directory + "/plots/" + case_name + "_losses.png",
    )

    # Plot latent space
    plotting.viz_3d_latent_space(
        model, x_test, dsd_time, output_directory + "/plots", case_name
    )

    # TODO: wasserstein & other metrics across all test members

    # Plot distributions: reconstruction
    r_bins_edges = ds_all["mass_bin"]
    test_ids = [0, 10, 20, 25]
    plotting.plot_reconstructions(
        model,
        test_ids,
        x_test,
        r_bins_edges,
        saveas=output_directory + "/plots/" + case_name + "_reconstructions.png",
    )

    # Predictions: Multi time step
    tplt = [3, 5, 8, 12]  # [0, 1, 2, 3, 5, 8, 12]
    plotting.plot_predictions_dzdt(
        test_ids,
        tplt,
        params["latent_dim"],
        model,
        dsd_time,
        x_test,
        m_test,
        x_train,
        m_train,
        r_bins_edges,
        # saveas=output_directory + "/plots/" + case_name + "_predictions.png",
    )

    # Plot trajectories of the latent variables
    plotting.plot_latent_trajectories_dzdt(
        params["latent_dim"],
        model,
        x_test,
        m_test,
        test_data.dt,
        test_data.t,
        x_train,
        m_train,
        plt_dx=False,
        # saveas=output_directory + "/plots/" + case_name + "_trajectories.png",
    )
