import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from training_scripts import eval_models_testdata as eval
from src import models, data_utils as du

if __name__ == "__main__":
    model_type = "SINDy"  # AR, NNdzdt, SINDy
    which_data = "val"  # "val", "9600", "14400", "RICO"
    print(f"Loading {model_type} model...")

    # load model & params
    model = eval.get_model(model_type, n_bins=64)
    model_dir = Path(eval.MODEL_DIRS[model_type])
    model_files = list(model_dir.glob(f"*.pth"))
    if not model_files:
        raise FileNotFoundError(
            f"No model files found in {eval.MODEL_DIRS[model_type]}"
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
