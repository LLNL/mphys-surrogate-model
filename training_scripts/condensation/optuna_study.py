"""
Script to conduct hyperparameter optimization with Optuna for the condensation model
"""

import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import src.data_utils as du

# MODEL_TYPE = "AE-AR"  # Don't uncomment for now
MODEL_TYPE = "NNdzdt"  # Don't uncomment for now
# MODEL_TYPE = "AE-SINDy"

# AE-AR model not implemented for condensation yet
if MODEL_TYPE == "AE-AR":
    raise NotImplementedError(
        f"Model type {MODEL_TYPE} not implemented for condensation dataset yet"
    )

if MODEL_TYPE == "AE-AR":
    from training_scripts.coalescence.train_ae_ar import (
        AEAutoregressor,
        params,
        train_and_eval,
    )
elif MODEL_TYPE == "NNdzdt":
    from training_scripts.condensation.train_ae_nndzdt import (
        AENNdzdtThermo,
        params,
        train_and_eval,
        get_ae_nndzdt_preds,
    )
elif MODEL_TYPE == "AE-SINDy":
    from training_scripts.condensation.train_ae_sindy import (
        AESINDyThermo,
        params,
        train_and_eval,
        get_ae_sindy_preds,
    )
else:
    raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

MAX_EPOCHS = 100


def objective(trial, params, n_bins, train_data, test_data, max_epochs=MAX_EPOCHS):
    # Set seed
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    random.seed(params["random_seed"])

    # Hyperparameter options
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2**i for i in range(2, 10)])
    if MODEL_TYPE == "AE-AR":
        layer1_size = trial.suggest_int("layer1_size", 20, 180)
        layer2_size = trial.suggest_int("layer2_size", 20, 180)
        layer3_size = trial.suggest_int("layer3_size", 20, 180)
        w_dx = trial.suggest_float("w_dx", 0.1, 1.9)
        w_dz = trial.suggest_float("w_dz", 0.1, 1.9)
    elif MODEL_TYPE == "NNdzdt":
        num_layers = trial.suggest_int("num_layers", 2, 7)
        layers = []
        for i in range(num_layers):
            ls = trial.suggest_int(f"layer{i}_size", 10, 200)
            layers.append(ls)
        # lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
        # lambda_S = trial.suggest_float("lambda_S", 1e-5, 1e0, log=True)
        lambda_x = trial.suggest_float("lambda_x", 1e-6, 1e1, log=True)
        lambda_z = trial.suggest_float("lambda_z", 1e-6, 1e1, log=True)
        lambda_S = trial.suggest_float("lambda_S", 1e-6, 1e1, log=True)
        lambda_r = 1.0
    elif MODEL_TYPE == "AE-SINDy":
        # # Latent dim and poly order
        # latent_dim = trial.suggest_int("latent_dim", 1, 4)
        # poly_order = trial.suggest_int("poly_order", 2, 3)
        # params["latent_dim"] = latent_dim
        # params["poly_order"] = poly_order
        # lambda1_metaweight = params["lambda1_metaweight"]
        # lambda_S = params["loss_weight_sindy_S"]

        # Weights
        lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
        lambda_S = trial.suggest_float("lambda_S", 0.1, 2.0)
        # lambda_x = trial.suggest_float("lambda_x", 1e-6, 1e1, log=True)
        # lambda_z = trial.suggest_float("lambda_z", 1e-6, 1e1, log=True)
        # lambda_S = trial.suggest_float("lambda_S", 1e-6, 1e1, log=True)
        # lambda_r = 1.0
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Initialize the model
    if MODEL_TYPE == "AE-AR":
        params["w_dx"] = w_dx
        params["w_dz"] = w_dz
        model = AEAutoregressor(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            layer_size=(layer1_size, layer2_size, layer3_size),
            n_lag=params["n_lag"],
        )
    elif MODEL_TYPE == "NNdzdt":
        # lambda1, lambda2, lambda3 = du.champion_calculate_weights(
        #     train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        # )
        params["loss_weight_recon"] = lambda_r
        params["loss_weight_x"] = lambda_x
        params["loss_weight_z"] = lambda_z
        params["loss_weight_S"] = lambda_S
        model = AENNdzdtThermo(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            layer_sizes=layers,
        )
    elif MODEL_TYPE == "AE-SINDy":
        lambda_x, lambda_z, lambda_r = du.champion_calculate_weights(
            train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        )
        params["loss_weight_sindy_x"] = lambda_x
        params["loss_weight_sindy_z"] = lambda_z
        params["loss_weight_recon"] = lambda_r
        params["loss_weight_sindy_S"] = lambda_S
        model = AESINDyThermo(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            poly_order=params["poly_order"],
            n_thermo=train_data.n_thermo,
        )
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=params["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # Training loop
    train_output = train_and_eval(
        max_epochs,
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        params,
        early_stopping=None,
        print_flag=False,
        optuna_trial=trial,
    )
    best_model = train_output[0]

    # Calculate error (RMSE) over full dataset
    if MODEL_TYPE == "AE-AR":
        raise NotImplementedError(
            f"Model type {MODEL_TYPE} not implemented for condensation dataset yet"
        )
    elif MODEL_TYPE == "NNdzdt":
        _, _, _, train_pred_dx = get_ae_nndzdt_preds(train_data, best_model)
        tdx = train_data.dx.squeeze()
        pdx = train_pred_dx.detach().numpy().squeeze()
    elif MODEL_TYPE == "AE-SINDy":
        _, _, _, train_pred_dx = get_ae_sindy_preds(train_data, best_model)
        tdx = train_data.dx.squeeze()
        pdx = train_pred_dx.detach().numpy().squeeze()
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")
    rmse = np.sqrt(np.mean((pdx - tdx) ** 2))

    return rmse


def optimize_worker(args):
    worker_id, n_trials, storage_url, study_name, objective_func = args

    # Set this once per worker process
    torch.set_num_threads(1)

    # Do optimization
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective_func, n_trials=n_trials)

    return None


if __name__ == "__main__":
    n_startup_trials = 24
    total_trials = (
        n_startup_trials + 8
    )  # On mac with 8 perf. cores, choose multiple of 8 total_trials
    parallel_flag = True

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
    test_data = du.NormedBinThermoDatasetDzDt(
        data["x_test"],
        data["dx_test"],
        data["thermo_test"],
        data["dthermo_test"],
    )

    # Set up save folder
    base_output_directory = Path("../../results/Optuna/")
    id = str(uuid.uuid4().hex)
    output_directory = base_output_directory / (
        f"{MODEL_TYPE}_" + datetime.now().isoformat().split(".")[0]  # + "_" + id
    )
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{output_directory}' already exists.")

    # Set up parallel info
    n_workers = 8  # 8 performance cores and 4 efficiency cores
    if parallel_flag:
        if total_trials % n_workers:
            raise RuntimeError("Ensure total trials is a multiple of n_workers")
    trials_per_worker = int(total_trials / n_workers)

    # Set up SQLite storage in results folder
    db_path = output_directory / "study.db"
    storage_url = f"sqlite:///{db_path}"
    study_name = MODEL_TYPE

    # Set up study
    # pruner = optuna.pruners.HyperbandPruner(
    #     min_resource=5,  # Don't consider pruning until epoch 5. With reduction_factor = 3, and max_resources = 100, will consider pruning at epochs 5, 15, 45, 100
    #     max_resource=MAX_EPOCHS,  # maximum epochs per trial
    #     reduction_factor=3,  # default, controls bracket sizes)
    # )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,  # don't prune at all for first 10 trials
        n_warmup_steps=30,  # don't prune any trial before epoch 30
        interval_steps=5,  # check every 5 epochs after warmup
    )
    tpe_sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,  # pure random exploration for first trials
        seed=params["random_seed"],
        multivariate=True,  # Experimental, may want to remove
    )
    study = optuna.create_study(
        storage=storage_url,
        sampler=tpe_sampler,
        pruner=pruner,
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )

    # Run study
    start_time = time.time()
    objective_with_args = partial(
        objective,
        params=params,
        n_bins=n_bins,
        train_data=train_data,
        test_data=test_data,
    )
    if parallel_flag:
        worker_args = [
            (i, trials_per_worker, storage_url, study_name, objective_with_args)
            for i in range(n_workers)
        ]
        with Pool(processes=n_workers) as pool:
            results = pool.map(optimize_worker, worker_args)
    else:
        study.optimize(objective_with_args, n_trials=total_trials)
    stop_time = time.time()

    # Save best hyperparameters
    best_params_file = output_directory / "best_params.json"
    best_params = {
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "optuna_runtime": stop_time - start_time,
        "random_seed": params["random_seed"],
    }
    best_params_file.write_text(json.dumps(best_params, indent=4))

    # Save full study object
    study_file = output_directory / "study.pkl"
    torch.save(study, study_file)

    # Save visualizations
    try:
        import optuna.visualization as vis

        vis.plot_optimization_history(study).write_html(
            str(output_directory / "opt_history.html")
        )
        vis.plot_param_importances(study).write_html(
            str(output_directory / "param_importance.html")
        )
        vis.plot_parallel_coordinate(study).write_html(
            str(output_directory / "parallel_coords.html")
        )
        vis.plot_contour(study).write_html(
            str(output_directory / "param_contours.html")
        )

    except Exception as e:
        print(f"Could not generate visualizations: {e}")

    # Save all trials to CSV
    csv_file = output_directory / "all_trials.csv"
    res_frame = study.trials_dataframe()
    res_frame.to_csv(csv_file)

    # Print best value
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Duration: {stop_time - start_time}")
