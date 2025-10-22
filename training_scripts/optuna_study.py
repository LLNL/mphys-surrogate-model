"""
Script to conduct hyperparameter optimization with Optuna for each of the three models
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import src.data_utils as du
from src import diagnostics

MODEL_TYPE = "AE-AR"
# MODEL_TYPE = "NNdzdt"
# MODEL_TYPE = "AE-SINDy"

if MODEL_TYPE == "AE-AR":
    from training_scripts.train_ae_ar import AEAutoregressor, params, train_and_eval
elif MODEL_TYPE == "NNdzdt":
    from training_scripts.train_ae_NNdzdt import AENNdzdt, params, train_and_eval
elif MODEL_TYPE == "AE-SINDy":
    from training_scripts.train_ae_sindy import AESINDy, params, train_and_eval
else:
    raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")


def objective(trial, params, n_bins, train_data, test_data, dsd_time, x_train, m_train):
    # Set seed
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    random.seed(params["random_seed"])

    # Hyperparameter options
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    if MODEL_TYPE == "AE-AR":
        layer1_size = trial.suggest_int("layer1_size", 20, 180)
        layer2_size = trial.suggest_int("layer2_size", 20, 180)
        layer3_size = trial.suggest_int("layer3_size", 20, 180)
        w_dx = trial.suggest_float("w_dx", 0.1, 1.9)
        w_dz = trial.suggest_float("w_dz", 0.1, 1.9)
    elif MODEL_TYPE == "NNdzdt":
        layer1_size = trial.suggest_int("layer1_size", 20, 60)
        layer2_size = trial.suggest_int("layer2_size", 20, 60)
        layer3_size = trial.suggest_int("layer3_size", 20, 60)
        lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
    elif MODEL_TYPE == "AE-SINDy":
        # latent_dim = trial.suggest_int("latent_dim", 1, 4)
        # poly_order = trial.suggest_int("poly_order", 2, 3)
        lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Fixed parameters
    num_epochs = 30  # Reduced for faster trials

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
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        )
        params["loss_weight_recon"] = lambda3
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2
        model = AENNdzdt(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            layer_size=(layer1_size, layer2_size, layer3_size),
        )
    elif MODEL_TYPE == "AE-SINDy":
        # params["latent_dim"] = latent_dim
        # params["poly_order"] = poly_order
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        )
        params["loss_weight_recon"] = lambda3
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2
        model = AESINDy(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            poly_order=params["poly_order"],
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
        num_epochs,
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

    # Calculate error (wass distance) over full dataset
    if MODEL_TYPE == "AE-AR":
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_AR(
            params["latent_dim"], best_model, dsd_time, x_train, m_train
        )
    elif MODEL_TYPE == "NNdzdt":
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_dzdt(
            params["latent_dim"],
            best_model,
            test_data.t,
            x_train,
            m_train,
            x_train,
            m_train,
        )
    elif MODEL_TYPE == "AE-SINDy":
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_dzdt(
            params["latent_dim"],
            best_model,
            test_data.t,
            x_train,
            m_train,
            x_train,
            m_train,
        )
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")
    _, train_wass, _, _ = diagnostics.get_performance_metrics(
        x_train, m_train, z_pred, x_pred
    )
    mean_trainset_wass = np.mean(train_wass)

    return mean_trainset_wass


def optimize_worker(args):
    worker_id, n_trials, storage_url, study_name, objective_func = args

    # Set this once per worker process
    torch.set_num_threads(1)

    # Do optimization
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective_func, n_trials=n_trials)

    return None


if __name__ == "__main__":
    total_trials = 1024  # On mac with 8 perf. cores, choose multiple of 8 total_trials
    parallel_flag = True

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

    # Set up datasets and loaders
    if MODEL_TYPE == "AE-AR":
        train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
        test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])
    elif MODEL_TYPE == "NNdzdt" or MODEL_TYPE == "AE-SINDy":
        train_data = du.NormedBinDatasetDzDt(x_train, dsd_time, m_train)
        test_data = du.NormedBinDatasetDzDt(x_test, dsd_time, m_test)
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Set weights
    if MODEL_TYPE == "NNdzdt" or MODEL_TYPE == "AE-SINDy":
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(train_data)
        params["loss_weight_recon"] = 1.0
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2

    # Set up save folder
    base_output_directory = Path("../results/Optuna/")
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
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        storage=storage_url,
        sampler=sampler,
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
        dsd_time=dsd_time,
        x_train=x_train,
        m_train=m_train,
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
    with csv_file.open("w", newline="") as f:
        fieldnames = ["trial_number", "value", "state", "duration"] + list(
            study.best_trial.params.keys()
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "duration": str(trial.duration),
            }
            row.update(trial.params)
            writer.writerow(row)

    # Print best value
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Duration: {stop_time - start_time}")
