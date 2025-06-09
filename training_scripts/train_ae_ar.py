import sys
import os
import time
import copy
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch
import pickle as pkl
import uuid
from src import data_utils as du, models, training, plotting

params = {
    "data_src": "box",
    "random_seed": 10,
    "num_epochs": 5,
    "batch_size": 512,
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
    "layer_size": (100, 100, 100),
    "CNN": False,
    "print_frequency": 1,
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
test_ids = [0, 10, 20, 25]
tplt = [0, 30, -1]
if params["CNN"]:
    prefix = params["data_src"] + "_CNN"
else:
    prefix = params["data_src"] + "_FFNN"


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class AEAutoregressor(torch.nn.Module):
    def __init__(
        self, n_channels=2, n_bins=100, n_latent=10, layer_size=None, n_lag=1, CNN=False
    ):
        super(AEAutoregressor, self).__init__()

        self.n_lag = n_lag
        if CNN:
            self.encoder = models.CNNEncoder(
                n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
            )
            self.decoder = models.CNNDecoder(
                n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
            )

        else:
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
    early_stopping,
    print_flag=False,
):
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
        for batch_X, batch_y, batch_M in train_loader:
            # Forward pass
            pred_y = model(batch_X, batch_M)  # DSD pred t+1
            pred_z = model.encoder(batch_X)  # Latent pred
            pred_z1 = model.autoregressor(
                torch.cat(
                    (
                        pred_z.reshape(-1, 1, params["latent_dim"] * params["n_lag"]),
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
                torch.log(pred_y + params["tol"]),
                torch.log(batch_y + params["tol"]),
            )
            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_X + params["tol"]),
            )
            loss = (
                params["w_dx"] * loss_dx
                + params["w_recon"] * loss_recon
                + params["w_dz"] * loss_dz
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

        # Test
        model.eval()
        for batch_X, batch_y, batch_M in test_loader:
            # Forward pass
            pred_y = model(batch_X, batch_M)  # DSD pred t+1
            pred_z = model.encoder(batch_X)  # Latent pred
            pred_z1 = model.autoregressor(
                torch.cat(
                    (
                        pred_z.reshape(-1, 1, params["latent_dim"] * params["n_lag"]),
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
                torch.log(pred_y + params["tol"]),
                torch.log(batch_y + params["tol"]),
            )
            loss_recon = divergence(
                torch.log(pred_x_recon + params["tol"]),
                torch.log(batch_X + params["tol"]),
            )

            loss = (
                params["w_dx"] * loss_dx
                + params["w_recon"] * loss_recon
                + params["w_dz"] * loss_dz
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
        if epoch % 1 == 0:
            if print_flag:
                print(
                    f"Epoch {epoch}/{n_epochs} | Train Loss: {losses[epoch]:.4f} | Test Loss: {test_losses[epoch]:.4f} | LR: {scheduler.get_last_lr()[0]:.4e} | Epoch Time: {epoch_end_time - epoch_start_time} s"
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
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # torch.backends.cudnn.benchmark = True
    print(
        f"Using {device} device"
    )  # TODO: while device code is here, I don't think the device is actually being used

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
        ) = du.open_erf_dataset()
    else:
        raise NotImplementedError("only erf and box data options exist")

    train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=x_test.shape[0], shuffle=True
    )

    # Initialize the model
    model = AEAutoregressor(
        n_channels=1,
        n_bins=n_bins,
        n_latent=params["latent_dim"],
        n_lag=params["n_lag"],
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
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Save and plot
    # ------------------------------------------------------------------------------------------------------------------
    best_model.eval()
    output_directory = "../trained_models/ae_ar_normed"
    id = uuid.uuid4().hex
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

    # Save model
    torch.save(
        best_model.state_dict(), output_directory + "/model/" + case_name + ".pth"
    )
    print(f"Saved model and losses as {case_name}")

    # Loss plot
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
        saveas=output_directory + "/plots/" + case_name + "_losses.png",
    )

    # Plot distributions: reconstruction
    plotting.plot_reconstructions(
        model,
        test_ids,
        x_test,
        r_bins_edges,
        saveas=output_directory + "/plots/" + case_name + "_reconstructions.png",
    )

    # Predictions: Multi time step
    plotting.plot_predictions_AE_AR(
        model,
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
        model,
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
        case_name,
    )
