"""
Main script that defines AE-SINDy model and training
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import optuna
from matplotlib import pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle as pkl
import uuid

import numpy as np
import seaborn as sns
import torch
from src import data_utils as du
from src import diagnostics, models, plotting
from torch.utils.data import DataLoader

params = {
    "data_src": "erfCond",
    "random_seed": 10,
    "num_epochs": 1000,
    "batch_size": 4,
    "learning_rate": 0.003365166078721918,
    "latent_dim": 4,
    "poly_order": 3,
    "lr_sched": True,
    "patience": 20,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_metaweight": 1.301768397667494,
    "loss_weight_sindy_S": 1.0432026964488583,  # TODO: explore more
    "print_frequency": 1,
    "load_ae_from": None,
    # "load_ae_from": "../../results/Optuna/ERF Dataset/AE-SINDy_LimParams/erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc",
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
class AESINDyThermo(torch.nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_bins=100,
        n_latent=10,
        poly_order=2,
        n_thermo=3,
    ):
        super(AESINDyThermo, self).__init__()
        self.poly_order = poly_order
        self.n_thermo = n_thermo
        assert n_channels == 1
        self.encoder = models.FFNNEncoder(n_bins=n_bins, n_latent=n_latent)
        self.decoder = models.FFNNDecoder(
            n_bins=n_bins, n_latent=n_latent, distribution=True
        )
        self.dzdt = models.SINDyDeriv(
            n_latent=n_latent + n_thermo,
            poly_order=poly_order,
            use_thresholds=False,
        )

    def forward(self, bin0, thermo):
        z0 = self.encoder(bin0)
        dzMdt = self.dzdt(z0, thermo)
        return dzMdt


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
    dS_losses = np.zeros(n_epochs) * np.nan
    test_losses = np.zeros(n_epochs) * np.nan
    test_recon_losses = np.zeros(n_epochs) * np.nan
    test_dx_losses = np.zeros(n_epochs) * np.nan
    test_dz_losses = np.zeros(n_epochs) * np.nan
    test_dS_losses = np.zeros(n_epochs) * np.nan
    best_test_loss = float("inf")
    best_model = None

    for epoch in range(n_epochs):
        # Train
        epoch_start_time = time.time()
        model.train()
        mean_epoch_loss = [0, 0, 0, 0, 0]
        for batch_x, batch_dx, batch_S, batch_dS in train_loader:
            batch_x = batch_x.to(device)
            batch_dx = batch_dx.to(device)
            batch_S = batch_S.to(device)
            batch_dS = batch_dS.to(device)

            # Forward pass
            z = model.encoder(batch_x)
            pred_x_recon = model.decoder(z)
            pred_dz_tot = model.dzdt(
                z, batch_S
            )  # In z-space, what is the time derivative of the input? One thing to look at in parity plots
            pred_dz = pred_dz_tot[:, :, : -model.n_thermo]  # Z-space dsd component
            pred_dS = pred_dz_tot[
                :, :, -model.n_thermo :
            ]  # Z-space thermo component (physics/state variables)
            _, dz = torch.func.jvp(
                model.encoder, (batch_x,), (batch_dx,)
            )  # Projection of time derivatives through encoder. dz is dx passed through encoder. dz should be compared to pred_dz. "Truth"
            _, pred_dx = torch.func.jvp(
                model.decoder, (z,), (pred_dz,)
            )  # Decoded z space time derivative prediction
            # Anything with `pred` prefix involves SINDy, other vars (except pred_x_recon) is everything else

            # Calculate train loss
            loss_dz = criterion(pred_dz, dz)
            loss_dS = criterion(pred_dS, batch_dS)
            loss_dx = criterion(pred_dx, batch_dx)
            loss_recon = divergence(
                torch.log(pred_x_recon + parameters["tol"]),
                torch.log(batch_x + parameters["tol"]),
            )
            loss = (
                parameters["loss_weight_sindy_x"] * loss_dx
                + parameters["loss_weight_recon"] * loss_recon
                + parameters["loss_weight_sindy_z"] * loss_dz
                + parameters["loss_weight_sindy_S"] * loss_dS
            )

            mean_epoch_loss[0] += loss.item()
            mean_epoch_loss[1] += loss_recon.item()
            mean_epoch_loss[2] += loss_dx.item()
            mean_epoch_loss[3] += loss_dz.item()
            mean_epoch_loss[4] += loss_dS.item()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            optimizer.step()

        # Save train losses
        losses[epoch] = mean_epoch_loss[0] / len(train_loader)
        recon_losses[epoch] = mean_epoch_loss[1] / len(train_loader)
        dx_losses[epoch] = mean_epoch_loss[2] / len(train_loader)
        dz_losses[epoch] = mean_epoch_loss[3] / len(train_loader)
        dS_losses[epoch] = mean_epoch_loss[4] / len(train_loader)

        # Test
        model.eval()
        mean_test_loss = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for batch_x, batch_dx, batch_S, batch_dS in test_loader:
                batch_x = batch_x.to(device)
                batch_dx = batch_dx.to(device)
                batch_S = batch_S.to(device)
                batch_dS = batch_dS.to(device)

                # Forward pass
                pred_x_recon = model.decoder(model.encoder(batch_x))
                z = model.encoder(batch_x)
                pred_dz_tot = model.dzdt(z, batch_S)
                pred_dz = pred_dz_tot[:, :, : -model.n_thermo]
                pred_dS = pred_dz_tot[:, :, -model.n_thermo :]
                _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
                _, pred_dx = torch.func.jvp(model.decoder, (z,), (pred_dz,))

                # Calculate test loss
                loss_dz = criterion(pred_dz, dz)
                loss_dS = criterion(pred_dS, batch_dS)
                loss_dx = criterion(pred_dx, batch_dx)
                loss_recon = divergence(
                    torch.log(pred_x_recon + parameters["tol"]),
                    torch.log(batch_x + parameters["tol"]),
                )

                loss = (
                    parameters["loss_weight_sindy_x"] * loss_dx
                    + parameters["loss_weight_recon"] * loss_recon
                    + parameters["loss_weight_sindy_z"] * loss_dz
                    + parameters["loss_weight_sindy_S"] * loss_dS
                )

                mean_test_loss[0] += loss.item()
                mean_test_loss[1] += loss_recon.item()
                mean_test_loss[2] += loss_dx.item()
                mean_test_loss[3] += loss_dz.item()
                mean_test_loss[4] += loss_dS.item()

        # Save test losses
        test_losses[epoch] = mean_test_loss[0] / len(test_loader)
        test_recon_losses[epoch] = mean_test_loss[1] / len(test_loader)
        test_dx_losses[epoch] = mean_test_loss[2] / len(test_loader)
        test_dz_losses[epoch] = mean_test_loss[3] / len(test_loader)
        test_dS_losses[epoch] = mean_test_loss[4] / len(test_loader)

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
                f"Recon: {parameters['loss_weight_recon'] * recon_losses[epoch]:.4f} | "
                f"dx: {parameters['loss_weight_sindy_x'] * dx_losses[epoch]:.4f} | "
                f"dz: {parameters['loss_weight_sindy_z'] * dz_losses[epoch]:.4f} | "
                f"dS: {parameters['loss_weight_sindy_S'] * dS_losses[epoch]:.4f}"
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
        dS_losses,
        test_losses,
        test_recon_losses,
        test_dx_losses,
        test_dz_losses,
        test_dS_losses,
    )


def get_ae_sindy_preds(data, model):
    """
    Return sindy predictions

    Args:
        data: Data class created by `du.NormedBinThermoDatasetDzDt`
        model: Instance of AESINDyThermo model
        n_thermo: Number of thermodynamic variables

    Returns:

    """
    pred_dz_tot = model(torch.from_numpy(data.x), torch.from_numpy(data.S))
    pred_dz = pred_dz_tot[:, :, : -data.n_thermo]
    pred_dS = pred_dz_tot[:, :, -data.n_thermo :]
    true_dz = torch.func.jvp(
        model.encoder,
        (torch.from_numpy(data.x),),
        (torch.from_numpy(data.dx),),
    )[1]
    pred_dx = torch.func.jvp(
        model.decoder,
        (model.encoder(torch.from_numpy(data.x)),),
        (pred_dz,),
    )[1]
    return pred_dz, true_dz, pred_dS, pred_dx


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

    # Open dataset
    data = du.open_cond_dataset(
        "../../data/erf_data/congestus/cond_tendency_14400_200m_filtered.nc"
    )
    n_bins = data["n_bins"]

    train_data = du.NormedBinThermoDatasetDzDt(
        data["x_train"],
        data["dx_train"],
        data["thermo_train"],
        data["dthermo_train"],
    )
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    test_data = du.NormedBinThermoDatasetDzDt(
        data["x_test"],
        data["dx_test"],
        data["thermo_test"],
        data["dthermo_test"],
    )
    test_loader = DataLoader(
        test_data, batch_size=data["x_test"].shape[0], shuffle=True
    )

    # Initialize the model
    n_thermo = train_data.n_thermo
    model = AESINDyThermo(
        n_channels=1,
        n_bins=n_bins,
        n_latent=params["latent_dim"],
        poly_order=params["poly_order"],
        n_thermo=n_thermo,
    )

    if params["load_ae_from"] is not None:
        # Load the coalescence ROM
        ae_sindy = AESINDyThermo(
            n_channels=1,
            n_bins=n_bins,
            n_latent=3,
            poly_order=2,
            n_thermo=1,
        )
        model_dir = Path(params["load_ae_from"])
        model_files = list(model_dir.glob(f"*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model files found")
        ae_sindy.load_state_dict(torch.load(model_files[0], weights_only=True))

        # transfer and fix the autoencoder parameters
        model.encoder = ae_sindy.encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.decoder = ae_sindy.decoder
        for param in model.decoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
        model.decoder.eval()

    # Optimizer and scheduling
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params["learning_rate"],
        weight_decay=params["wd"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stopping = diagnostics.EarlyStopping(patience=params["patience"])

    total_params = sum(p.numel() for p in model.parameters())
    total_coeffs = sum(p.numel() for p in model.dzdt.parameters())
    print(
        f"Total number of parameters: {total_params}, {total_coeffs} are SINDy coefficients"
    )

    # Compute & set weights based on Champion et al recs
    lambda_x, lambda_z, _ = du.champion_calculate_weights(
        train_data, lambda1_metaweight=params["lambda1_metaweight"]
    )
    params["loss_weight_recon"] = 1.0
    params["loss_weight_sindy_x"] = lambda_x
    params["loss_weight_sindy_z"] = lambda_z

    # Training loop
    # ----------------------------------------------------------------------------------
    (
        best_model,
        losses,
        recon_losses,
        dx_losses,
        dz_losses,
        dS_losses,
        test_losses,
        test_recon_losses,
        test_dx_losses,
        test_dz_losses,
        test_dS_losses,
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
    case_name = prefix + "_latent{}_order{}_tr{}_lr{}_bs{}_weights{}-{}-{}_{}".format(
        params["latent_dim"],
        params["poly_order"],
        params["num_epochs"],
        params["learning_rate"],
        params["batch_size"],
        params["loss_weight_recon"],
        params["loss_weight_sindy_z"],
        params["loss_weight_sindy_x"],
        params["loss_weight_sindy_S"],
        id,
    )
    print(f"Save ID is {case_name}")

    # Set save dir
    runsp_out_dir = Path("../../ng_scripts/trained_models/ae_SINDy") / case_name
    if not runsp_out_dir.exists():
        runsp_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output files to {runsp_out_dir}*")

    # Save losses
    pkl_out_files = []
    pkl_out_files.append(runsp_out_dir / (case_name + ".pkl"))
    for out_file in pkl_out_files:
        with open(out_file, "wb") as pickle_file:
            pkl.dump(
                (
                    losses,
                    recon_losses,
                    dx_losses,
                    dz_losses,
                    dS_losses,
                    test_losses,
                    test_recon_losses,
                    test_dx_losses,
                    test_dz_losses,
                    test_dS_losses,
                ),
                pickle_file,
            )

    # Save params
    params_out_files = []
    params_out_files.append(runsp_out_dir / (case_name + "_params.json"))
    params_save = copy.deepcopy(params)
    for key, value in params_save.items():
        if type(value) is np.float32 or type(value) is np.float64:
            params_save[key] = float(value)
    for out_file in params_out_files:
        out_file.write_text(json.dumps(params_save, indent=4))

    # Save model
    mdl_out_files = []
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
            params["loss_weight_sindy_S"] * np.array(dS_losses),
            params["loss_weight_recon"] * np.array(recon_losses),
        ],
        labels=["dx/dt", "dz/dt", "dS/dt", "Recon"],
        title=f"Training Loss",
    )
    fig.savefig(runsp_out_dir / (case_name + "_losses.png"))
    fig.show()

    # Plot distributions: reconstruction
    fig = plotting.plot_reconstructions(
        best_model,
        test_ids,
        data["x_test"],
        data["r_bins_edges"],
    )
    fig.savefig(runsp_out_dir / (case_name + "_reconstructions.png"))
    fig.show()

    # Make predictions using test data to compare
    test_pred_dz, test_true_dz, test_pred_dS, test_pred_dx = get_ae_sindy_preds(
        test_data, best_model
    )

    # Plot thermo variables
    fig, axes = plt.subplots(
        nrows=1, ncols=n_thermo, figsize=(28, 10), layout="constrained"
    )
    for idx, ax in enumerate(axes.flatten()):
        true = test_data.dSdt[:, :, idx].ravel()
        pred = test_pred_dS[:, :, idx].detach().numpy().ravel()
        all_data = np.concatenate((true, pred), axis=0)
        lb, ub = 0.95 * np.min(all_data), 1.05 * np.max(all_data)
        if np.allclose(true, 0.0):
            ax.text(
                0.5,
                0.5,
                "N/A",
                transform=ax.transAxes,
                fontsize=22,
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="gray", alpha=0.75),
            )
        else:
            ax.scatter(
                true, pred, s=15, c="tab:blue", marker=".", alpha=0.90, label="Data"
            )
            ax.axline(
                (0.0, 0.0),
                (1.0, 1.0),
                color="tab:red",
                linestyle="--",
                alpha=0.8,
                label="1:1 Line",
            )
            ax.set_xlim([lb, ub])
            ax.set_ylim([lb, ub])
            ax.legend()
        ax.grid(linestyle=":", color="black", alpha=0.25)
        ax.set_aspect("equal")
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Thermodynamic Variable {idx+1}/{n_thermo}")
    fig.savefig(runsp_out_dir / (case_name + "_dSdt_Parity.png"))
    fig.show()

    # Plot z-space
    fig, axes = plt.subplots(
        nrows=1, ncols=params["latent_dim"], figsize=(28, 10), layout="constrained"
    )
    for idx, ax in enumerate(axes.flatten()):
        true = test_true_dz[:, :, idx].detach().numpy().ravel()
        pred = test_pred_dz[:, :, idx].detach().numpy().ravel()
        all_data = np.concatenate((true, pred), axis=0)
        lb, ub = 0.95 * np.min(all_data), 1.05 * np.max(all_data)
        ax.scatter(true, pred, s=15, c="tab:blue", marker=".", alpha=0.90, label="Data")
        ax.axline(
            (0.0, 0.0),
            (1.0, 1.0),
            color="tab:red",
            linestyle="--",
            alpha=0.8,
            label="1:1 Line",
        )
        ax.set_xlim([lb, ub])
        ax.set_ylim([lb, ub])
        ax.legend()
        ax.grid(linestyle=":", color="black", alpha=0.25)
        ax.set_aspect("equal")
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Z-Space Derivatives {idx+1}/{params['latent_dim']}")
    fig.savefig(runsp_out_dir / (case_name + "_dz_Parity.png"))

    # Convert dx into distributions by normalizing
    tdx = test_data.dx.squeeze()
    pdx = test_pred_dx.detach().numpy().squeeze()
    all_data = np.concatenate((tdx, pdx), axis=0)
    vmin = np.quantile(all_data, q=0.01)
    vmax = np.quantile(all_data, q=0.99)

    # Plot distribution comparison
    fig, axes = plt.subplots(2, 1, figsize=(34, 13), layout="constrained")
    cbar_kwargs = {"fraction": 0.046, "pad": 0.01}
    cmap = "viridis"
    # ---
    ax = axes[0]
    sns.heatmap(
        tdx.T,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar_kws=cbar_kwargs,
        ax=ax,
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Bin")
    ax.set_title(f"True dx")
    # ---
    ax = axes[1]
    sns.heatmap(
        pdx.T,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar_kws=cbar_kwargs,
        ax=ax,
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Bin")
    ax.set_title(f"Predicted dx")
    # ---
    fig.savefig(runsp_out_dir / (case_name + "_dx_Comparison.png"))
    fig.show()


# - TODO: predictions stepped forward in time
