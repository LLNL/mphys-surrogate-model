"""
Test script to verify unified encoder/decoder interface works with all combinations.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(project_root)

import numpy as np
import torch

from src import model_factory, recon_coalescence_losses, training_utils

# Test combinations
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

    # Mixed encoder/decoder
    {"encoder": "ffnn", "decoder": "nwi_simple", "dynamics": "sindy", "name": "FFNN-NWI-SINDy"},
    {"encoder": "nwi", "decoder": "ffnn", "dynamics": "sindy", "name": "NWI-FFNN-SINDy"},
]

def run_test(config):
    """Run one epoch of training with given configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")

    params = {
        "encoder_type": config["encoder"],
        "decoder_type": config["decoder"],
        "dynamics_type": config["dynamics"],
        "num_blocks": 3,
        "hidden_size": 128,
        "poly_order": 2,
        "layer_size": (100, 100, 100),
        "n_lag": 1,
        "data_src": "erf",
        "random_seed": 42,
        "num_epochs": 1,
        "batch_size": 25,
        "learning_rate": 0.001,
        "wd": 1e-3,
        "latent_dim": 3,
        "tol": 1e-8,
        "lambda1_metaweight": 0.5,
        "loss_weight_l2": 1.0,
    }

    try:
        # Setup
        torch.manual_seed(params["random_seed"])
        np.random.seed(params["random_seed"])

        device = training_utils.setup_device(params)
        print(f"Device: {device}")

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

        print(f"Model created successfully")
        print(f"  Encoder output shape test: ", end="")

        # Test encoder output shape
        batch = next(iter(train_loader))
        batch_x = batch[0].to(device)
        Z = model.encoder(batch_x)
        expected_shape = (batch_x.shape[0], batch_x.shape[1], params["latent_dim"] + 1)
        assert Z.shape == expected_shape, f"Expected {expected_shape}, got {Z.shape}"
        print(f"✓ {Z.shape}")

        # Test decoder output shape
        print(f"  Decoder output shape test: ", end="")
        x_recon = model.decoder(Z)
        assert x_recon.shape == batch_x.shape, f"Expected {batch_x.shape}, got {x_recon.shape}"
        print(f"✓ {x_recon.shape}")

        # Get loss function
        loss_fn = recon_coalescence_losses.get_loss_function(params["dynamics_type"])

        # Get training dataset for loss weight computation
        train_dataset = train_loader.dataset

        # Compute loss weights if needed
        if params["dynamics_type"] in ["sindy", "nn_dzdt"]:
            params = training_utils.setup_loss_weights(params, train_dataset)
        elif params["dynamics_type"] == "autoregressive":
            if "w_dx" not in params:
                params["w_dx"] = 1.0
                params["w_recon"] = 1.0
                params["w_dz"] = 0.01

        # Run one training step
        print(f"  Training step test: ", end="")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
        )

        model.train()
        batch = [b.to(device) for b in batch]

        optimizer.zero_grad()
        loss, loss_dict = loss_fn(model, batch, params, device)
        loss.backward()
        optimizer.step()

        print(f"✓ Loss: {loss.item():.6f}")
        print(f"    Loss breakdown: {', '.join([f'{k}: {v.item():.6f}' for k, v in loss_dict.items() if k != 'total'])}")

        # Test evaluation
        print(f"  Evaluation test: ", end="")
        model.eval()
        with torch.no_grad():
            test_batch = [b.to(device) for b in next(iter(test_loader))]
            test_loss, test_loss_dict = loss_fn(model, test_batch, params, device)
        print(f"✓ Test loss: {test_loss.item():.6f}")

        # Test plotting for all model types (matching save_utils.generate_plots)
        print(f"  Plot generation test: ", end="")
        from src import plotting, diagnostics
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing

        test_ids = [0, 1]  # Just 2 samples for testing
        tplt = [0, -1]  # First and last time points

        # Test reconstruction plot (works for all model types)
        fig = plotting.plot_reconstructions(
            model,
            test_ids,
            metadata["x_test"],
            metadata["r_bins_edges"],
        )
        assert fig is not None
        print(f"✓ Reconstructions", end="")

        # Test dynamics-specific plots
        if params["dynamics_type"] in ["sindy", "nn_dzdt"]:
            # Test predictions plot
            fig = plotting.plot_predictions_dzdt(
                test_ids,
                tplt,
                params["latent_dim"],
                model,
                metadata["dsd_time"],
                metadata["x_test"],
                metadata["m_test"],
                metadata["x_train"],
                metadata["m_train"],
                metadata["r_bins_edges"],
            )
            assert fig is not None
            print(f", Predictions", end="")

            # Test latent trajectories
            z_pred, z_data, _ = diagnostics.get_latent_trajectories_dzdt(
                params["latent_dim"],
                model,
                metadata["dsd_time"],
                metadata["x_test"][:5],
                metadata["m_test"][:5],
                metadata["x_train"][:5],
                metadata["m_train"][:5],
            )
            fig = plotting.plot_latent_trajectories(
                params["latent_dim"],
                metadata["dsd_time"],
                z_pred,
                z_data,
            )
            assert fig is not None
            print(f", Latent trajectories", end="")

        elif params["dynamics_type"] == "autoregressive":
            # AR-specific plots
            # Test AR predictions plot
            fig = plotting.plot_predictions_AE_AR(
                model,
                test_ids,
                metadata["dsd_time"],
                tplt,
                metadata["x_test"],
                metadata["m_test"],
                metadata["r_bins_edges"],
                n_lag=params["n_lag"]
            )
            assert fig is not None
            print(f", Predictions", end="")

            # Test AR latent trajectories
            z_pred, z_data, _ = diagnostics.get_latent_trajectories_AR(
                params["latent_dim"],
                model,
                metadata["dsd_time"],
                metadata["x_test"][:5],
                metadata["m_test"][:5],
                n_lag=params["n_lag"]
            )
            fig = plotting.plot_latent_trajectories(
                params["latent_dim"],
                metadata["dsd_time"],
                z_pred,
                z_data,
            )
            assert fig is not None
            print(f", Latent trajectories", end="")

        # Test 3D latent space visualization (generic for all models)
        fig = plotting.viz_3d_latent_space(
            model,
            metadata["x_test"][:10],  # Just 10 samples
            metadata["dsd_time"],
        )
        assert fig is not None
        print(f", 3D latent space", end="")

        # Test NWI weights plot if applicable
        if params["encoder_type"] == "nwi":
            fig = plotting.plot_nnwi_weights(model, metadata["r_bins_edges"])
            assert fig is not None
            print(f", NWI weights", end="")

        print(f" ✓")

        print(f"\n✅ {config['name']} PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {config['name']} FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing unified encoder/decoder interface")
    print(f"Total configurations to test: {len(test_configs)}")

    results = {}
    for config in test_configs:
        results[config["name"]] = run_test(config)

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
