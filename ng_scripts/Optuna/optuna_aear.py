import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

import src.data_utils as du
from training_scripts.train_ae_ar import AEAutoregressor, params, train_and_eval


def objective(trial):
    # Set seed
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    random.seed(params["random_seed"])

    # Hyperparameter tuning
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)

    # Fixed parameters
    num_epochs = 10  # Reduced for faster trials

    # Initialize the model
    model = AEAutoregressor(
        n_channels=1,
        n_bins=n_bins,
        n_latent=params["latent_dim"],
        n_lag=params["n_lag"],
        CNN=params["CNN"],
    )

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
        early_stopping=None,
        print_flag=False,
        optuna_trial=trial,
    )
    losses = train_output[1]
    best_train_loss = np.min(losses)

    return best_train_loss


if __name__ == "__main__":
    # Open dataset
    start_time = time.time()
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
    train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
    test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])

    # Set up save folder
    base_output_directory = Path("./")
    id = str(uuid.uuid4().hex)
    output_directory = base_output_directory / (
        "AE-AR_" + datetime.now().isoformat().split(".")[0] + "_" + id
    )
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{output_directory}' already exists.")

    # Set up SQLite storage in results folder
    db_path = output_directory / "study.db"
    storage_url = f"sqlite:///{db_path}"

    # Set up and run study
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        study_name="AE-AR",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)

    # Save best hyperparameters
    best_params_file = output_directory / "best_params.json"
    best_params = {"value": study.best_trial.value, "params": study.best_trial.params}
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
    stop_time = time.time()
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Duration: {stop_time - start_time}")
