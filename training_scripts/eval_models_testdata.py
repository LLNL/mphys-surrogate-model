"""
Evaluate the three trained models for various data and test sizes all at once
"""

import json
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch
from src import data_utils as du
from src import diagnostics, plotting

from train_ae_ar import AEAutoregressor
from train_ae_NNdzdt import AENNdzdt
from train_ae_sindy import AESINDy

MODEL_DIRS = {
    "AR": "../results/Optuna/ERF Dataset/AE-AR_2025-09-18T10:18:57_PostARBugfix/erf_FFNN_latent3_order(141, 154, 40)_tr1000_lr0.0030348411572892766_bs8_weights0.12789450188986579-1.1729901013704414_a0f49326688d4e69bcc0e9a78da3c870",
    "SINDy": "../results/Optuna/ERF Dataset/AE-SINDy_LimParams/erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc",
    "NNdzdt": "../results/Optuna/ERF Dataset/NNdzdt_2025-07-20T23:31:20_3a400c596947422389559813cd41dfe6/erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3",
}
DATA_DIRS = {
    "val": "../data/congestus_coal_200m_test.nc",
    "9600": "../data/erf_data/congestus/noadv_coal_200m_9600.nc",
    "14400": "../data/erf_data/congestus/noadv_coal_200m_14400.nc",
    "RICO": "../data/erf_data/RICO/noadv_coal_200m.nc",
    "train": "../data/congestus_coal_200m_train.nc",
}
TEST_SIZES = {
    "val": 0.99,
    "9600": 0.2,
    "14400": 0.02,
    "RICO": 0.1,
    "train": 0.01,
}


def load_best_params(model_type):
    load_dir = MODEL_DIRS[model_type]
    with open(os.path.join(load_dir, "../best_params.json")) as f:
        best_params = json.load(f)
    params = best_params["params"]
    return params


def get_model(model_type, n_bins=64):
    params = load_best_params(model_type)
    if model_type == "AR":
        model = AEAutoregressor(
            n_channels=1,
            n_bins=n_bins,
            n_latent=3,
            n_lag=1,
            layer_size=(
                params["layer1_size"],
                params["layer2_size"],
                params["layer3_size"],
            ),
        )
    elif model_type == "SINDy":
        model = AESINDy(
            n_channels=1,
            n_bins=n_bins,
            n_latent=3,
            poly_order=2,
        )
    elif model_type == "NNdzdt":
        model = AENNdzdt(
            n_channels=1,
            n_bins=n_bins,
            n_latent=3,
            layer_size=(
                params["layer1_size"],
                params["layer2_size"],
                params["layer3_size"],
            ),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


if __name__ == "__main__":
    which_data = "val"  # "val", "9600", "14400", "RICO"
    for model_type in ["AR", "SINDy", "NNdzdt"]:
        for which_data in ["val", "9600", "14400", "RICO"]:
            print(f"Loading {model_type} model...")

            # load model & params
            model = get_model(model_type, n_bins=64)
            model_dir = Path(MODEL_DIRS[model_type])
            model_files = list(model_dir.glob(f"*.pth"))
            if not model_files:
                raise FileNotFoundError(
                    f"No model files found in {MODEL_DIRS[model_type]}"
                )
            model.load_state_dict(torch.load(model_files[0], weights_only=True))
            model.eval()

            # load data
            print(f"Evaluating {which_data} data...")
            train_data = du.open_mass_dataset(
                "_",
                "_",
                filepath=DATA_DIRS["train"],
                test_size=TEST_SIZES["train"],
                sample_time=np.arange(0, 61, 5),
            )
            data_pth = DATA_DIRS[which_data]
            data = du.open_mass_dataset(
                "_",
                "_",
                filepath=data_pth,
                test_size=TEST_SIZES[which_data],
                sample_time=np.arange(0, 61, 5),
                m_scale=train_data["m_scale"],
            )
            x_test = data["x_test"]
            m_test = data["m_test"]
            x_train = train_data["x_train"]
            m_train = train_data["m_train"]
            r_bins_edges = data["r_bins_edges"]
            dsd_time = data["dsd_time"]

            # Tests
            if model_type == "AR":
                z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_AR(
                    3, model, dsd_time, x_test, m_test
                )
            else:
                z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_dzdt(
                    3,
                    model,
                    dsd_time,
                    x_test,
                    m_test,
                    x_train,
                    m_train,
                )
            (
                test_kl,
                test_wass,
                test_wun,
                test_mass_diff,
            ) = diagnostics.get_performance_metrics(x_test, m_test, z_pred, x_pred)

            # plot full testset performance
            fig = plotting.plot_full_testset_performance_pred(
                test_kl, test_wass, test_mass_diff
            )
            fig.show()

            # Plot quantiles from test set
            tplt = [0, 5, -1]
            fig = plotting.plot_testset_quantiles_pred(
                x_test, x_pred, test_wass, tplt, dsd_time, r_bins_edges
            )
            fig.show()
