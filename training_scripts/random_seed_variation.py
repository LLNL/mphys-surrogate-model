"""
Script to test random seed variation for each of the three models
"""

import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import src.data_utils as du

# MODEL_TYPE = "AE-AR"
# MODEL_TYPE = "NNdzdt"
MODEL_TYPE = "AE-SINDy"

if MODEL_TYPE == "AE-AR":
    from training_scripts.train_ae_ar import AEAutoregressor, params, train_and_eval
elif MODEL_TYPE == "NNdzdt":
    from training_scripts.train_ae_NNdzdt import AENNdzdt, params, train_and_eval
elif MODEL_TYPE == "AE-SINDy":
    from training_scripts.train_ae_sindy import AESINDy, params, train_and_eval
else:
    raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")


def train_model(args):
    # Unpack args
    random_seed, n_bins, train_loader, test_loader, params = args

    # Set this once per worker process
    torch.set_num_threads(1)

    # Set seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Initialize the model
    if MODEL_TYPE == "AE-AR":
        model = AEAutoregressor(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            n_lag=params["n_lag"],
            CNN=params["CNN"],
        )
    elif MODEL_TYPE == "NNdzdt":
        model = AENNdzdt(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            layer_size=params["layer_size"],
            CNN=params["CNN"],
        )
    elif MODEL_TYPE == "AE-SINDy":
        model = AESINDy(
            n_channels=1,
            n_bins=n_bins,
            n_latent=params["latent_dim"],
            poly_order=params["poly_order"],
            CNN=params["CNN"],
        )
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # Training loop
    num_epochs = 10
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
        optuna_trial=None,
    )
    losses = train_output[1]
    best_train_loss = np.min(losses)

    # Print
    print(f"Best train loss for seed {random_seed}: {best_train_loss}")

    return best_train_loss


if __name__ == "__main__":
    total_trials = 200  # On mac with 8 perf. cores, choose multiple of 8 total_trials
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
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # Set weights
    if MODEL_TYPE == "NNdzdt" or MODEL_TYPE == "AE-SINDy":
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(train_data)
        params["loss_weight_recon"] = 1.0
        params["loss_weight_sindy_x"] = lambda1
        params["loss_weight_sindy_z"] = lambda2

    # Set up save folder
    base_output_directory = Path("../ng_scripts/Random_Seed_Variation/")
    id = str(uuid.uuid4().hex)
    output_directory = base_output_directory / (
        f"{MODEL_TYPE}_" + datetime.now().isoformat().split(".")[0] + "_" + id
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

    # Run study
    start_time = time.time()
    if parallel_flag:
        worker_args = [
            (i, n_bins, train_loader, test_loader, params) for i in range(total_trials)
        ]
        with Pool(processes=n_workers) as pool:
            res = pool.map(train_model, worker_args)
        res = np.array(res)
    else:
        res = np.zeros(total_trials)
        for i in range(total_trials):
            args = (i, n_bins, train_loader, test_loader, params)
            res[i] = train_model(args)
    stop_time = time.time()
    print(f"Duration: {stop_time - start_time}")

    # Calculate
    best_seed = np.argmin(res)
    best_train_loss = res[best_seed]

    # Save best results
    best_seed_file = output_directory / "best_seed.json"
    best_seed_dict = {
        "loss": float(best_train_loss),
        "seed": int(best_seed),
        "runtime": stop_time - start_time,
    }
    best_seed_file.write_text(json.dumps(best_seed_dict, indent=4))

    # Save all results
    all_params_file = output_directory / "all_seeds.csv"
    sf = pd.Series(res, name="loss").sort_values()
    sf.index.name = "seed"
    sf.to_csv(all_params_file)

    # Plot results
    hist_file = output_directory / "hist.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), layout="constrained")
    # ---
    ax = axes[0]
    ax.hist(res, bins=20, label="Training Loss")
    ax.set_xlabel("Training Loss")
    ax.set_ylabel("Number of trials")
    ax.set_title("Total")
    # ---
    ax = axes[1]
    ax.hist(
        res, bins=20, range=(res.min(), np.quantile(res, 0.95)), label="Training Loss"
    )
    ax.set_xlabel("Training Loss")
    ax.set_ylabel("Number of trials")
    ax.set_title("95th Percentile Zoom")
    # ---
    fig.suptitle(f"Training Loss Over Random Seed ({total_trials} Trials)")
    fig.savefig(hist_file)

    # Print best values
    print(f"Best seed is {best_seed} with loss {best_train_loss}")

    # Finish up
    plt.close("all")
