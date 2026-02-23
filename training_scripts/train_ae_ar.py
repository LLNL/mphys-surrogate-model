"""
Main script that defines AE-AR model and training
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import optuna

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle as pkl
import uuid

import numpy as np
import torch
from src import data_utils as du
from src import diagnostics, models, plotting

params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0030348411572892766,
    "latent_dim": 3,
    "n_lag": 1,
    "w_recon": 1,
    "w_dx": 0.12789450188986579,
    "w_dz": 1.1729901013704414,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "layer_size": (141, 154, 40),
    "print_frequency": 1,
    "emily_save": False,
    "nipun_save": True,
}

# Global variables and settings
# Criterion and divergence need to be outside train function to be available in other scripts
torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
test_ids = [0, 10, 20, 30]
tplt = [0, 5, -1]
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class AEAutoregressor(torch.nn.Module):
    def __init__(self, n_channels=2, n_bins=100, n_latent=10, layer_size=None, n_lag=1):
        super(AEAutoregressor, self).__init__()
        self.n_lag = n_lag
        self.encoder = models.FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
        self.decoder = models.FFNNDecoder(
            n_bins=n_bins, n_latent=n_latent, distribution=True
        )
        self.autoregressor = models.Autoregressive(
            n_bins=n_latent + 1,
            n_bins_in=n_latent * self.n_lag + 1,
            layer_size=layer_size,
        )

    def forward(self, bin0, M):
        latent0 = []
        for t in range(self.n_lag):
            latent0.append(self.encoder(bin0[:, t, :]).unsqueeze(1))
        latent0 = torch.cat(latent0, dim=2)
        latent0_M = torch.cat([latent0, M], dim=2)
        latent1_M = self.autoregressor(latent0_M)
        latent1 = latent1_M[:, :, :-1]
        bin1 = self.decoder(latent1)
        return bin1


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
    parameters,
    early_stopping=None,
    print_flag=False,
    device="cpu",
    optuna_trial=None,
):
    model.to(device)
    # if device == "cpu":
    #     torch.set_num_threads(1)

    # Set up loss storage and other vars
    losses = np.zeros(n_epochs) * np.nan
    recon_losses = np.zeros(n_epochs) * np.nan
    dx_losses = np.zeros(n_epochs) * np.nan
    dz_losses = np.zeros(n_epochs) * np.nan
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
        for batch_X, batch_y, batch_M in train_loader:
            # Forward pass
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_M = batch_M.to(device)

            pred_y = model(batch_X, batch_M)  # DSD pred t+1
            pred_z = model.encoder(batch_X)  # Latent pred
            pred_z1 = model.autoregressor(
                torch.cat(
                    (
                        pred_z.reshape(
                            -1, 1, parameters["latent_dim"] * parameters["n_lag"]
                        ),
                        batch_M,
                    ),
                    dim=2,
                )
            )  # Latent t+1 pred
            data_z1 = torch.cat(
                (model.encoder(batch_y), batch_M), dim=2
            )  # Latent t+1 "actual"
            pred_x_recon = model.decoder(model.encoder(batch_X))  # DSD pred t+0

            # Calculate train loss
            loss_dz = criterion(pred_z1, data_z1)
            loss_dx = divergence(
                torch.log(pred_y + parameters["tol"]),
                torch.log(batch_y + parameters["tol"]),
            )
            loss_recon = divergence(
                torch.log(pred_x_recon + parameters["tol"]),
                torch.log(batch_X + parameters["tol"]),
            )
            loss = (
                parameters["w_dx"] * loss_dx
                + parameters["w_recon"] * loss_recon
                + parameters["w_dz"] * loss_dz
            )

            mean_epoch_loss[0] += loss.item()
            mean_epoch_loss[1] += loss_recon.item()
            mean_epoch_loss[2] += loss_dx.item()
            mean_epoch_loss[3] += loss_dz.item()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Save train losses
        losses[epoch] = mean_epoch_loss[0] / len(train_loader)
        recon_losses[epoch] = mean_epoch_loss[1] / len(train_loader)
        dx_losses[epoch] = mean_epoch_loss[2] / len(train_loader)
        dz_losses[epoch] = mean_epoch_loss[3] / len(train_loader)

        # Test
        model.eval()
        for batch_X, batch_y, batch_M in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_M = batch_M.to(device)

            # Forward pass
            pred_y = model(batch_X, batch_M)  # DSD pred t+1
            pred_z = model.encoder(batch_X)  # Latent pred
            pred_z1 = model.autoregressor(
                torch.cat(
                    (
                        pred_z.reshape(
                            -1, 1, parameters["latent_dim"] * parameters["n_lag"]
                        ),
                        batch_M,
                    ),
                    dim=2,
                )
            )  # Latent t+1 pred
            data_z1 = torch.cat(
                (model.encoder(batch_y), batch_M), dim=2
            )  # Latent t+1 "actual"
            pred_x_recon = model.decoder(model.encoder(batch_X))  # DSD pred t+0

            # Calculate test loss
            loss_dz = criterion(pred_z1, data_z1)
            loss_dx = divergence(
                torch.log(pred_y + parameters["tol"]),
                torch.log(batch_y + parameters["tol"]),
            )
            loss_recon = divergence(
                torch.log(pred_x_recon + parameters["tol"]),
                torch.log(batch_X + parameters["tol"]),
            )

            loss = (
                parameters["w_dx"] * loss_dx
                + parameters["w_recon"] * loss_recon
                + parameters["w_dz"] * loss_dz
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
        if parameters["lr_sched"]:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        # Print
        epoch_end_time = time.time()
        if epoch % parameters["print_frequency"] == 0 and print_flag:
            print(
                f"Epoch [{epoch}/{parameters['num_epochs']}], Train Loss: {losses[epoch]:.4f} | "
                f"Test Loss: {test_losses[epoch]:.4f} | LR: {scheduler.get_last_lr()}"
                f"| Epoch Time: {epoch_end_time - epoch_start_time} s"
            )
            print(
                f"Recon: {parameters['w_recon'] * recon_losses[epoch]:.4f} | "
                f"dx: {parameters['w_dx'] * dx_losses[epoch]:.4f} | "
                f"dz: {parameters['w_dz'] * dz_losses[epoch]:.4f} | "
            )

        # Optional optuna report
        if optuna_trial is not None:
            optuna_trial.report(losses[epoch], epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping
        if early_stopping is not None:
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
        else (
            "mps"
            if torch.backends.mps.is_available() and params["batch_size"] > 1000
            else "cpu"
        )
    )
    # torch.backends.cudnn.benchmark = True
    print(f"Using {device} device")

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

    train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # Initialize the model
    model = AEAutoregressor(
        n_channels=1,
        n_bins=n_bins,
        n_latent=params["latent_dim"],
        n_lag=params["n_lag"],
        layer_size=params["layer_size"],
    )

    # Optimizer and scheduling
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stopping = diagnostics.EarlyStopping(patience=params["patience"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

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
        params,
        early_stopping=early_stopping,
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
    prefix = params["data_src"] + "_FFNN"
    case_name = prefix + "_latent{}_order{}_tr{}_lr{}_bs{}_weights{}-{}_{}".format(
        params["latent_dim"],
        params["layer_size"],
        params["num_epochs"],
        params["learning_rate"],
        params["batch_size"],
        params["w_dx"],
        params["w_dz"],
        id,
    )
    print(f"Save ID is {case_name}")

    # Emily save dirs
    tpsp_out_dir = Path("../trained_models/ae_ar_normed")
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
    runsp_out_dir = Path("../ng_scripts/trained_models/ae_ar_normed") / case_name
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

    # Save params
    params_out_files = []
    if params["emily_save"]:
        params_out_files.append(tpsp_mod_dir / (case_name + "_params.json"))
    if params["nipun_save"]:
        params_out_files.append(runsp_out_dir / (case_name + "_params.json"))
    params_save = copy.deepcopy(params)
    for key, value in params_save.items():
        if type(value) is np.float32 or type(value) is np.float64:
            params_save[key] = float(value)
    for out_file in params_out_files:
        out_file.write_text(json.dumps(params, indent=4))

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
            params["w_dx"] * dx_losses,
            params["w_dz"] * dz_losses,
            params["w_recon"] * recon_losses,
        ],
        labels=["X: t -> t+1", "Z: t -> t+1", "Recon"],
        title=f"Training Loss, lag {params['n_lag']}",
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
    fig = plotting.plot_predictions_AE_AR(
        best_model,
        test_ids,
        dsd_time,
        tplt,
        x_test,
        m_test,
        r_bins_edges,
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_predictions.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_predictions.png"))

    # Plot trajectories of the latent variables
    z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_AR(
        params["latent_dim"], best_model, dsd_time, x_test, m_test
    )
    fig = plotting.plot_latent_trajectories(
        params["latent_dim"], dsd_time, z_pred, z_data
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

    test_kl, test_wass, test_wun, _ = diagnostics.get_performance_metrics(
        x_test, m_test, z_pred, x_pred
    )
    fig = plotting.plot_full_testset_performance_pred(test_kl, test_wass, test_wun)
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_full_test_pred.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_full_test_pred.png"))

    # Plot quantiles from test set
    fig = plotting.plot_testset_quantiles_pred(
        x_test, x_pred, test_wass, tplt, dsd_time, r_bins_edges
    )
    if params["emily_save"]:
        fig.savefig(tpsp_plot_dir / (case_name + "_quantiles_test_pred.png"))
    if params["nipun_save"]:
        fig.savefig(runsp_out_dir / (case_name + "_quantiles_test_pred.png"))

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
