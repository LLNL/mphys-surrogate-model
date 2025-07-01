import copy
import os
import sys
import time
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle as pkl
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import data_utils as du
from src import models, plotting, thresholding, training

params = {
    "data_src": "box",
    "random_seed": 10,
    "num_epochs": 1000,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    "CNN": False,
    "sequential_threshold_method": "bimodal_gmm",  # None, bimodal_gmm, knee_detection
    "sequential_thresholding_interval": 10,  # None
    "print_frequency": 1,
    "emily_save": True,
    "nipun_save": True,
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
test_ids = [0, 10, 20, 30]
tplt = [0, 5, -1]


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class AESINDy(torch.nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_bins=100,
        n_latent=10,
        poly_order=2,
        CNN=False,
        sequential_thresholding=False,
    ):
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
        self.dzdt = models.SINDyDeriv(
            n_latent=n_latent + 1,
            poly_order=poly_order,
            use_thresholds=sequential_thresholding,
        )

    def forward(self, bin0, M):
        z0 = self.encoder(bin0)
        dzMdt = self.dzdt(z0, M)
        dzdt = dzMdt[:, :, :-1]
        return dzdt


# ----------------------------------------------------------------------------------------------------------------------
# Training function
# ----------------------------------------------------------------------------------------------------------------------
def train_and_eval(
    n_epochs,
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    early_stopping,
    print_flag=False,
    device="cpu",
):
    model.to(device)

    # Set up loss storage and other vars
    losses = np.zeros(n_epochs) * np.nan
    recon_losses = np.zeros(n_epochs) * np.nan
    dx_losses = np.zeros(n_epochs) * np.nan
    dz_losses = np.zeros(n_epochs) * np.nan
    threshold_events = []
    test_losses = np.zeros(n_epochs) * np.nan
    test_recon_losses = np.zeros(n_epochs) * np.nan
    test_dx_losses = np.zeros(n_epochs) * np.nan
    test_dz_losses = np.zeros(n_epochs) * np.nan
    best_test_loss = float("inf")
    best_model = None

    for epoch in range(n_epochs):
        # Train
        epoch_start_time = time.time()
        model.train()
        mean_epoch_loss = [0, 0, 0, 0]
        for batch_x, batch_dx, batch_M in train_loader:
            batch_x = batch_x.to(device)
            batch_dx = batch_dx.to(device)
            batch_M = batch_M.to(device)

            # Forward pass
            pred_x_recon = model.decoder(model.encoder(batch_x))
            z = model.encoder(batch_x)
            zz = z.clone().detach().requires_grad_()
            pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
            _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
            _, pred_dx = torch.func.jvp(model.decoder, (zz,), (pred_dz,))

            # Calculate train loss
            loss_dz = criterion(pred_dz, dz)
            loss_dx = criterion(pred_dx, batch_dx)
            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_x + params["tol"]),
            )
            loss = (
                params["loss_weight_sindy_x"] * loss_dx
                + params["loss_weight_recon"] * loss_recon
                + params["loss_weight_sindy_z"] * loss_dz
            )

            mean_epoch_loss[0] += loss.item()
            mean_epoch_loss[1] += loss_recon.item()
            mean_epoch_loss[2] += loss_dx.item()
            mean_epoch_loss[3] += loss_dz.item()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Check for adaptive thresholding
        if params["sequential_threshold_method"] is not None:
            (
                thresholded,
                threshold,
                n_active,
                analysis,
            ) = thresholder.maybe_apply_threshold(epoch, loss.item())
        else:
            thresholded = False

        if thresholded:
            if print_flag:
                print(
                    f"Epoch {epoch}: Applied {analysis.get('threshold_method', 'unknown')} "
                    f"thresholding with threshold={threshold:.6f}, "
                    f"active coefficients={n_active} / {total_coeffs}"
                )
            threshold_events.append(epoch)
            coeffs = (
                model.dzdt.get_coeffs()
            )  # TODO: This var doesn't do anything, delete?

        # Save train losses
        losses[epoch] = mean_epoch_loss[0] / len(train_loader)
        recon_losses[epoch] = mean_epoch_loss[1] / len(train_loader)
        dx_losses[epoch] = mean_epoch_loss[2] / len(train_loader)
        dz_losses[epoch] = mean_epoch_loss[3] / len(train_loader)

        # Test
        model.eval()
        for batch_x, batch_dx, batch_M in test_loader:
            batch_x = batch_x.to(device)
            batch_dx = batch_dx.to(device)
            batch_M = batch_M.to(device)

            # Forward pass
            pred_x_recon = model.decoder(model.encoder(batch_x))
            z = model.encoder(batch_x)
            zz = z.clone().detach().requires_grad_()
            pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
            _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
            _, pred_dx = torch.func.jvp(model.decoder, (zz,), (pred_dz,))

            # Calculate test loss
            loss_dz = criterion(pred_dz, dz)
            loss_dx = criterion(
                pred_dx, batch_dx
            )  # TODO: Same as ae-nn, do we want divergence here?
            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_x + params["tol"]),
            )

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
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        # Print
        epoch_end_time = time.time()
        if epoch % params["print_frequency"] == 0 and print_flag:
            print(
                f"Epoch [{epoch}/{params['num_epochs']}], Train Loss: {losses[epoch]:.4f} | "
                f"Test Loss: {test_losses[epoch]:.4f} | LR: {scheduler.get_last_lr()}"
                f"| Epoch Time: {epoch_end_time - epoch_start_time} s"
            )
            print(
                f"Recon: {params['loss_weight_recon'] * recon_losses[epoch]:.4f} | "
                f"dx: {params['loss_weight_sindy_x'] * dx_losses[epoch]:.4f} | "
                f"dz: {params['loss_weight_sindy_z'] * dz_losses[epoch]:.4f} | "
            )

        # Early stopping
        early_stopping(loss)
        if early_stopping.early_stop:
            if print_flag:
                print("Training stopped early.")
            break

    return (
        best_model,
        losses,
        recon_losses,
        dx_losses,
        dz_losses,
        test_losses,
        test_recon_losses,
        test_dx_losses,
        test_dz_losses,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        # NOTE: Attemping to use MPS backend will lead to an error
        # "...as_strided_tensorimpl does not work with MPS..."
    )
    # torch.backends.cudnn.benchmark = True
    print(f"Using {device} device")

    start_time = time.time()
    # Open dataset
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
        ) = du.open_erf_dataset(sample_time=np.arange(0, 61, 5))
    else:
        raise NotImplementedError("only erf and box data options exist")

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
        sequential_thresholding=(
            True if params["sequential_thresholding_interval"] is not None else False
        ),
    )

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    if params["sequential_threshold_method"] is not None:
        thresholder = thresholding.AdaptiveSequentialThresholdingSINDy(
            model.dzdt,
            thresholding.AdaptiveThresholdAnalyzer(
                method=params["sequential_threshold_method"],
                min_epochs_between=params["sequential_thresholding_interval"],
            ),
        )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stopping = training.EarlyStopping(patience=params["patience"])

    total_params = sum(p.numel() for p in model.parameters())
    total_coeffs = sum(p.numel() for p in model.dzdt.parameters())
    print(
        f"Total number of parameters: {total_params}, {total_coeffs} are SINDy coefficients"
    )

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

    # Training loop
    # ----------------------------------------------------------------------------------
    (
        best_model,
        losses,
        recon_losses,
        dx_losses,
        dz_losses,
        test_losses,
        test_recon_losses,
        test_dx_losses,
        test_dz_losses,
    ) = train_and_eval(
        params["num_epochs"],
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        early_stopping,
        print_flag=True,
        device=device,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Save and plot
    # ------------------------------------------------------------------------------------------------------------------
    # Set up case name
    best_model.eval()
    best_model = best_model.to("cpu")
    id = str(uuid.uuid4().hex)
    if params["CNN"]:
        prefix = params["data_src"] + "_CNN"
    else:
        prefix = params["data_src"] + "_FFNN"
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
    print(f"Save ID is {case_name}")

    # Emily save dirs
    tpsp_out_dir = Path("../trained_models/ae_SINDy")
    if not tpsp_out_dir.exists():
        tpsp_out_dir.mkdir(parents=True, exist_ok=True)
    if not (tpsp_loss_dir := tpsp_out_dir / "losses").exists():
        tpsp_loss_dir.mkdir(parents=True, exist_ok=True)
    if not (tpsp_mod_dir := tpsp_out_dir / "models").exists():
        tpsp_mod_dir.mkdir(parents=True, exist_ok=True)
    if not (tpsp_plot_dir := tpsp_out_dir / "plots").exists():
        tpsp_plot_dir.mkdir(parents=True, exist_ok=True)
    if params["emily_save"]:
        print(
            f"Saving output files to the respective folders at {tpsp_out_dir}/{{losses,models,plots}}/{case_name}*"
        )

    # Nipun save dirs
    runsp_out_dir = Path("../ng_scripts/trained_models/ae_SINDy") / case_name
    if not runsp_out_dir.exists():
        runsp_out_dir.mkdir(parents=True, exist_ok=True)
    if params["nipun_save"]:
        print(f"Saving output files to {runsp_out_dir}*")

    # Save losses
    pkl_out_files = []
    if params["emily_save"]:
        pkl_out_files.append(tpsp_loss_dir / (case_name + ".pkl"))
    if params["nipun_save"]:
        pkl_out_files.append(runsp_out_dir / (case_name + ".pkl"))
    for out_file in pkl_out_files:
        with open(out_file, "wb") as pickle_file:
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

    # Save model
    mdl_out_files = []
    if params["emily_save"]:
        mdl_out_files.append(tpsp_mod_dir / (case_name + ".pth"))
    if params["nipun_save"]:
        mdl_out_files.append(runsp_out_dir / (case_name + ".pth"))
    for out_file in mdl_out_files:
        torch.save(best_model.state_dict(), out_file)

    # Loss plot
    fig = plotting.plot_losses(
        losses,
        test_losses=test_losses,
        sub_losses=[
            params["loss_weight_sindy_x"] * np.array(dx_losses),
            params["loss_weight_sindy_z"] * np.array(dz_losses),
            params["loss_weight_recon"] * np.array(recon_losses),
        ],
        labels=["dx/dt", "dz/dt", "Recon"],
        title=f"Training Loss",
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_losses.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_losses.png"))

    # Plot distributions: reconstruction
    fig = plotting.plot_reconstructions(
        best_model,
        test_ids,
        x_test,
        r_bins_edges,
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_reconstructions.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_reconstructions.png"))

    # Predictions: Multi time step
    fig = plotting.plot_predictions_dzdt(
        test_ids,
        tplt,
        params["latent_dim"],
        best_model,
        dsd_time,
        x_test,
        m_test,
        x_train,
        m_train,
        r_bins_edges,
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_predictions.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_predictions.png"))

    # Plot trajectories of the latent variables
    fig, z_pred = plotting.plot_latent_trajectories_dzdt(
        params["latent_dim"],
        best_model,
        x_test,
        m_test,
        test_data.dt,
        test_data.t,
        x_train,
        m_train,
        plt_dx=False,
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_trajectories.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_trajectories.png"))

    # Plot full test set performance
    fig = plotting.plot_full_testset_performance_recon(
        best_model, x_test, params["tol"]
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_full_test_recon.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_full_test_recon.png"))

    fig = plotting.plot_full_testset_performance_pred(
        best_model, x_test, z_pred, params["tol"]
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_full_test_pred.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_full_test_pred.png"))

    # Plot latent space
    fig = plotting.viz_3d_latent_space(
        best_model,
        x_test,
        dsd_time,
    )
    if params["emily_save"]:
        fig.write_html(tpsp_plot_dir / (case_name + "_latent_space.html"))
    if params["nipun_save"]:
        fig.write_html(runsp_out_dir / (case_name + "_latent_space.html"))
