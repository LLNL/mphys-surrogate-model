"""
Unified training script for all coalescence surrogate models.
Supports: AE-SINDy, AE-NNdzdt, AE-AR, and pure autoencoder (NNWI).

See docs/TRAINING_PARAMS.md for full documentation of all parameters.
"""

import os
import sys

import data_utils

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch

from src import model_factory, recon_coalescence_losses, save_utils, training_utils

# Parameters - configure these for your desired model
params = {
    # Model architecture
    "encoder_type": "nwi",  # "ffnn" or "nwi"
    "decoder_type": "nwi_simple",  # "ffnn", "nwi_simple", or "nwi_deep"
    "dynamics_type": "nn_dzdt",  # "sindy", "nn_dzdt", "autoregressive", or "none"

    # NWI-specific (only used if encoder_type="nwi" or decoder_type contains "nwi")
    "num_blocks": 3,
    "hidden_size": 128,
    # SINDy-specific (only used if dynamics_type="sindy")
    "poly_order": 2,
    # NN dzdt-specific (only used if dynamics_type="nn_dzdt")
    "layer_size": (100, 100, 100),
    # AR-specific (only used if dynamics_type="autoregressive")
    "n_lag": 1,

    # Data
    "data_src": "erf",  # "box" or "erf"

    # Training
    "random_seed": 10,
    "num_epochs": 4,
    "batch_size": 25,
    "learning_rate": 1e-3,
    "wd": 1e-3,
    "lr_sched": True,
    "patience": 50,
    "print_frequency": 1,

    # Model parameters
    "latent_dim": 3,

    # Loss parameters
    "tol": 1e-8,

    # Loss weight computation (for dzdt models - only used if weights not specified)
    "lambda1_metaweight": 0.5,
    "w_recon_vt": 0.0,
    # Optional: Manually specify loss weights (overrides Champion et al. computation)
    # For sindy/nn_dzdt: "loss_weight_recon", "loss_weight_dx", "loss_weight_dz", "loss_weight_vt_recon"
    # For none: "loss_weight_l2"
    # Output

    "save": False,
    "show_plots": False,
}

if __name__ == "__main__":
    # Setup
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    device = training_utils.setup_device()
    print(f"Using {device} device")
    print(
        f"Training sedimentation model model: {params['encoder_type']}/{params['decoder_type']}/{params['dynamics_type']}"
    )

    # Load data
    data = data_utils.open_sed_datasets()

    train_loader, test_loader, metadata = training_utils.setup_dataloaders_sed(
        params["data_src"], params
    )

    # Create model using factory
    model = model_factory.create_model(
        params["encoder_type"],
        params["decoder_type"],
        params["dynamics_type"],
        params,
        metadata["n_bins"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    if model.dzdt is not None:
        dynamics_params = sum(p.numel() for p in model.dzdt.parameters())
        print(
            f"Total parameters: {total_params}, {dynamics_params} in dynamics module"
        )
    else:
        print(f"Total parameters: {total_params} (pure autoencoder, no dynamics)")

    for batch in train_loader:
        batch_x, batch_flux, batch_m = batch
        W = model.encoder.wf_mat()
        h = model.encoder(batch_x)
        x_recon = model.decoder(h)
    # # Setup loss weights
    # if params["dynamics_type"] in ["sindy", "nn_dzdt"]:
    #     from src.data_utils import NormedBinDatasetDzDt
    #
    #     train_data = NormedBinDatasetDzDt(
    #         metadata["x_train"], metadata["dsd_time"], metadata["m_train"]
    #     )
    # elif params["dynamics_type"] == "autoregressive":
    #     from src.data_utils import NormedBinDatasetAR
    #
    #     train_data = NormedBinDatasetAR(
    #         metadata["x_train"], metadata["m_train"], lag=params["n_lag"]
    #     )
    # else:
    #     from src.data_utils import NormedBinDatasetDzDt
    #
    #     train_data = NormedBinDatasetDzDt(
    #         metadata["x_train"], metadata["dsd_time"], metadata["m_train"]
    #     )
    #
    # params = training_utils.setup_loss_weights(params, train_data)
    #
    # # Get loss function
    # loss_fn = recon_coalescence_losses.get_loss_function(params["dynamics_type"])
    #
    # # Setup optimization
    # optimizer, scheduler, early_stopping = training_utils.setup_optimization(
    #     model, params
    # )
    #
    # # Train
    # best_model, losses = training_utils.train_and_eval(
    #     model,
    #     train_loader,
    #     test_loader,
    #     optimizer,
    #     scheduler,
    #     loss_fn,
    #     params,
    #     device,
    #     early_stopping=early_stopping,
    # )
    #
    # # Save and plot
    # output_dir, case_name, timestamp = save_utils.setup_output_dir(params)
    # print(f"Output directory: {output_dir}")
    #
    # if params["save"]:
    #     save_utils.save_model_artifacts(
    #         best_model, losses, params, output_dir, timestamp
    #     )
    #
    #     # Plot losses
    #     save_utils.plot_training_losses(losses, params, output_dir)
    #
    #     # Generate all other plots
    #     save_utils.generate_plots(best_model, metadata, params, output_dir)
    #
    # print("Training complete!")
