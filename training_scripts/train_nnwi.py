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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle as pkl
import uuid

import numpy as np
import torch
from src import data_utils as du
from src import diagnostics, nwi, plotting
from torch.utils.data import DataLoader

params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 50,
    "batch_size": 25,
    "learning_rate": 1e-3,
    "latent_dim": 2,
    "num_blocks": 3,
    "hidden_size": 128,
    "lr_sched": True,
    "patience": 10,
    "tol": 1e-12,
    "wd": 1e-3,
    "loss_weight_l2": 1000.0,
    "print_frequency": 1,
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

    # Set up loss storage and other vars
    losses = np.zeros(n_epochs) * np.nan
    kl_losses = np.zeros(n_epochs) * np.nan
    l2_losses = np.zeros(n_epochs) * np.nan

    test_losses = np.zeros(n_epochs) * np.nan
    test_kl_losses = np.zeros(n_epochs) * np.nan
    test_l2_losses = np.zeros(n_epochs) * np.nan

    best_test_loss = float("inf")
    best_model = None

    for epoch in range(n_epochs):
        # Train
        epoch_start_time = time.time()
        model.train()
        mean_epoch_loss = [0, 0, 0]
        for batch_x, _, _ in train_loader:
            batch_x = batch_x.to(device)

            # Forward pass
            pred_x_recon = model.decoder(model.encoder(batch_x))

            # Calculate train loss
            loss_kl = divergence(
                torch.log(pred_x_recon + parameters["tol"]),
                torch.log(batch_x + parameters["tol"]),
            )
            loss_l2 = criterion(
                pred_x_recon, batch_x
            )
            loss = (
                loss_kl +
                parameters["loss_weight_l2"] * loss_l2
            )

            mean_epoch_loss[0] += loss.item()
            mean_epoch_loss[1] += loss_kl.item()
            mean_epoch_loss[2] += loss_l2.item()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Save train losses
        losses[epoch] = mean_epoch_loss[0] / len(train_loader)
        kl_losses[epoch] = mean_epoch_loss[1] / len(train_loader)
        l2_losses[epoch] = mean_epoch_loss[2] / len(train_loader)

        # Test
        model.eval()
        with torch.no_grad():
            for batch_x, _, _ in test_loader:
                batch_x = batch_x.to(device)

                # Forward pass
                pred_x_recon = model.decoder(model.encoder(batch_x))

                # Calculate test loss
                loss_kl = divergence(
                    torch.log(pred_x_recon + parameters["tol"]),
                    torch.log(batch_x + parameters["tol"]),
                )
                loss_l2 = criterion(
                    pred_x_recon, batch_x
                )
                loss = (
                        loss_kl +
                        parameters["loss_weight_l2"] * loss_l2
                )

        # Save test losses
        test_losses[epoch] = loss.item()
        test_kl_losses[epoch] = loss_kl.item()
        test_l2_losses[epoch] = loss_l2.item()

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
                f"KL: {kl_losses[epoch]:.4f} | "
                f"L2: {parameters['loss_weight_l2'] * l2_losses[epoch]:.4f} | "
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
        kl_losses,
        l2_losses,
        test_losses,
        test_kl_losses,
        test_l2_losses,
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
    enc = nwi.SimpleNWIEncoder(n_bins=64,
                               n_latent=params["latent_dim"],
                            )
    dec = nwi.SimpleDecoder(n_bins=64,
                               n_latent=params["latent_dim"],
                               hidden_features=params["hidden_size"],
                               num_blocks=params["num_blocks"]
                        )
    model = nwi.NNWIAutoencoder(enc, dec)

    for x_train, _, _ in train_loader:
        break

    # Optimizer and scheduling
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stopping = diagnostics.EarlyStopping(patience=params["patience"])

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Total number of parameters: {total_params}"
    )

    # Training loop
    # ----------------------------------------------------------------------------------
    (
        best_model,
        losses,
        kl_losses,
        l2_losses,
        test_losses,
        test_kl_losses,
        test_l2_losses,
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
    prefix = params["data_src"] + "_Simple"
    case_name = prefix + "_latent{}_hs{}_nb{}_tr{}_lr{}_bs{}_l2w{}_{}".format(
        params["latent_dim"],
        params["hidden_size"],
        params["num_blocks"],
        params["num_epochs"],
        params["learning_rate"],
        params["batch_size"],
        params["loss_weight_l2"],
        id,
    )
    print(f"Save ID is {case_name}")

    runsp_out_dir = Path("../trained_models/NNWI") / case_name
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
                    kl_losses,
                    l2_losses,
                    test_losses,
                    test_kl_losses,
                    test_l2_losses,
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
            np.array(kl_losses),
            params["loss_weight_l2"] * np.array(l2_losses),
        ],
        labels=["KL", "L2"],
        title=f"Training Loss",
    )
    fig.show()
    fig.savefig(runsp_out_dir / (case_name + "_losses.png"))

    # Plot distributions: reconstruction
    fig = plotting.plot_reconstructions(
        best_model,
        test_ids,
        x_test,
        r_bins_edges,
    )
    fig.show()
    fig.savefig(runsp_out_dir / (case_name + "_reconstructions.png"))


    # Plot full test set performance
    fig = plotting.plot_full_testset_performance_recon(
        best_model, x_test, params["tol"]
    )
    fig.show()
    fig.savefig(runsp_out_dir / (case_name + "_full_test_recon.png"))


    # Plot latent space
    fig = plotting.viz_3d_latent_space(
        best_model,
        x_test,
        dsd_time,
    )
    fig.write_html(runsp_out_dir / (case_name + "_latent_space.html"))
