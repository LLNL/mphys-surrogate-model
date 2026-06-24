"""
Unified training script for all coalescence surrogate models.
Supports: AE-SINDy, AE-NNdzdt, AE-AR, and pure autoencoder (NNWI).

See docs/TRAINING_PARAMS.md for full documentation of all parameters.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import numpy as np
import torch

from src import data_utils, model_factory, recon_sedimentation_losses, save_utils, training_utils, plotting

# Parameters - configure these for your desired model
params = {
    # Process type
    "process": "sedimentation", #"sedimentation" or "coalescence"

    # Model architecture
    "encoder_type": "nwi",  # "ffnn" or "nwi"
    "decoder_type": "nwi_simple",  # "ffnn", "nwi_simple", or "nwi_deep"
    "dynamics_type": "none",  # "sindy", "nn_dzdt", "autoregressive", or "none"

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
    "data_src": "erf",  # "erf_mini" or "erf"

    # Training
    "random_seed": 1,
    "num_epochs": 250,
    "batch_size": 1000,
    "learning_rate": 1e-3,
    "wd": 1e-3,
    "lr_sched": True,
    "patience": 50,
    "print_frequency": 1,

    # Model parameters
    "latent_dim": 3,

    # Loss parameters
    "tol": 1e-8,

    # Loss weight computation #TODO: Automate for sedimentation case?
    "loss_weight_recon": 1.0,
    "loss_weight_recon_vt": 0.0,
    "loss_weight_dx": 1e5,
    "loss_weight_dz": 1e1,
    "loss_weight_negFlux": 1e4,
    # Optional: Manually specify loss weights (overrides Champion et al. computation)
    # For sindy/nn_dzdt: "loss_weight_recon", "loss_weight_dx", "loss_weight_dz", "loss_weight_vt_recon"
    # For none: "loss_weight_l2"
    # Output

    "save": True,
    "plot": True,
    "show_plots": True,
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


    # Get loss function
    loss_fn = recon_sedimentation_losses.get_loss_function(params["dynamics_type"])

    # Setup optimization
    optimizer, scheduler, early_stopping = training_utils.setup_optimization(
        model, params
    )

    # Train
    best_model, losses = training_utils.train_and_eval(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_fn,
        params,
        device,
        early_stopping=early_stopping,
    )

    print("Training complete! Saving and plotting...")

    # Save and plot
    output_dir, case_name, timestamp = save_utils.setup_output_dir(params)
    print(f"Output directory: {output_dir}")

    if params["save"]:
        save_utils.save_model_artifacts(
            best_model, losses, params, output_dir, timestamp
        )

    if params["plot"]:

        # Plot losses
        save_utils.plot_training_losses(losses, params, output_dir)

        # Plot reconstructions
        test_ids = np.random.randint(0, data["x_test"].shape[0], 5)
        fig = plotting.plot_reconstructions(best_model, test_ids, data["x_test"][:,np.newaxis,:], data["r_bins_edges"])
        fig.savefig(output_dir / "reconstructions.png") if params["save"] else None
        fig.show() if params["show_plots"] else None

        # Plot NWI weights if applicable
        if params['encoder_type'] == "nwi":
            fig = plotting.plot_nnwi_weights(best_model, data['r_bins_edges'])
            fig.savefig(output_dir / "weights.png") if params['save'] else None
            fig.show() if params['show_plots'] else None

        # Plot flux reconstructions
        if params["dynamics_type"] != "none":
            fig = plotting.plot_flux_projections(best_model, test_ids, data["x_test"], data["flux_test"], data["r_bins_edges"])
            fig.savefig(output_dir / "projections.png") if params["save"] else None
            fig.show() if params["show_plots"] else None

            # Plot latent fluxes
            fig = plotting.plot_latent_fluxes(best_model, data["x_test"], data["flux_test"])
            fig.savefig(output_dir / "latent_fluxes.png") if params["save"] else None
            fig.show() if params["show_plots"] else None