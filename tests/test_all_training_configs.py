"""
Test script to verify all training configurations work end-to-end.
Tests all combinations of encoder/decoder/dynamics types that are used in practice.
"""

import os
import sys
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch

from src import model_factory, recon_coalescence_losses, save_utils, training_utils

# Test configurations - realistic combinations used in practice
test_configs = [
    # FFNN encoder/decoder with different dynamics
    {"encoder": "ffnn", "decoder": "ffnn", "dynamics": "sindy", "name": "FFNN-FFNN-SINDy"},
    {"encoder": "ffnn", "decoder": "ffnn", "dynamics": "nn_dzdt", "name": "FFNN-FFNN-NNdzdt"},
    {"encoder": "ffnn", "decoder": "ffnn", "dynamics": "autoregressive", "name": "FFNN-FFNN-AR"},
    {"encoder": "ffnn", "decoder": "ffnn", "dynamics": "none", "name": "FFNN-FFNN-AE"},

    # NWI encoder/decoder with different dynamics
    {"encoder": "nwi", "decoder": "nwi_simple", "dynamics": "sindy", "name": "NWI-NWI-SINDy"},
    {"encoder": "nwi", "decoder": "nwi_simple", "dynamics": "nn_dzdt", "name": "NWI-NWI-NNdzdt"},
    {"encoder": "nwi", "decoder": "nwi_simple", "dynamics": "autoregressive", "name": "NWI-NWI-AR"},
    {"encoder": "nwi", "decoder": "nwi_simple", "dynamics": "none", "name": "NWI-NWI-AE"},
]

def run_full_training_test(config):
    """Run full training pipeline with given configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")

    params = {
        # Model architecture
        "encoder_type": config["encoder"],
        "decoder_type": config["decoder"],
        "dynamics_type": config["dynamics"],

        # NWI-specific
        "num_blocks": 3,
        "hidden_size": 128,
        # SINDy-specific
        "poly_order": 2,
        # NN dzdt-specific
        "layer_size": (100, 100, 100),
        # AR-specific
        "n_lag": 1,

        # Data
        "data_src": "erf",

        # Training
        "random_seed": 42,
        "num_epochs": 1,
        "batch_size": 25,
        "learning_rate": 0.001,
        "wd": 1e-3,
        "lr_sched": False,
        "patience": 50,
        "print_frequency": 1,

        # Model parameters
        "latent_dim": 3,

        # Loss parameters
        "tol": 1e-8,
        "lambda1_metaweight": 0.5,

        # Output
        "save": True,
        "show_plots": False,
    }

    try:
        # Setup
        torch.manual_seed(params["random_seed"])
        np.random.seed(params["random_seed"])

        device = training_utils.setup_device()
        print(f"Using {device} device")

        # Load data
        train_loader, test_loader, metadata = training_utils.setup_dataloaders(
            params["data_src"], params
        )

        # Create model
        model = model_factory.create_model(
            params["encoder_type"],
            params["decoder_type"],
            params["dynamics_type"],
            params,
            metadata["n_bins"],
        )
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        # Setup loss weights
        if params["dynamics_type"] in ["sindy", "nn_dzdt"]:
            from src.data_utils import NormedBinDatasetDzDt
            train_data = NormedBinDatasetDzDt(
                metadata["x_train"], metadata["dsd_time"], metadata["m_train"]
            )
        elif params["dynamics_type"] == "autoregressive":
            from src.data_utils import NormedBinDatasetAR
            train_data = NormedBinDatasetAR(
                metadata["x_train"], metadata["m_train"], lag=params["n_lag"]
            )
        else:
            from src.data_utils import NormedBinDatasetDzDt
            train_data = NormedBinDatasetDzDt(
                metadata["x_train"], metadata["dsd_time"], metadata["m_train"]
            )

        params = training_utils.setup_loss_weights(params, train_data)

        # Get loss function
        loss_fn = recon_coalescence_losses.get_loss_function(params["dynamics_type"])

        # Setup optimization
        optimizer, scheduler, early_stopping = training_utils.setup_optimization(
            model, params
        )

        # Train
        print("Training...")
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

        # Save and plot
        output_dir, case_name, timestamp = save_utils.setup_output_dir(params)
        print(f"Output directory: {output_dir}")

        if params["save"]:
            print("Saving artifacts...")
            save_utils.save_model_artifacts(
                best_model, losses, params, output_dir, timestamp
            )

            # Plot losses
            save_utils.plot_training_losses(losses, params, output_dir)

            # Generate all other plots
            print("Generating plots...")
            save_utils.generate_plots(best_model, metadata, params, output_dir)

        # Clean up the trained model to save space
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleaned up test artifacts from {output_dir}")

        print(f"\n✅ {config['name']} PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {config['name']} FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing full training pipeline for all configurations")
    print(f"Total configurations to test: {len(test_configs)}")

    results = {}
    for config in test_configs:
        results[config["name"]] = run_full_training_test(config)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        sys.exit(1)
