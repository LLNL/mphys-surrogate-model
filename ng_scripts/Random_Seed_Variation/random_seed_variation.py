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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

import src.data_utils as du

MODEL_TYPE = "AE-AR"
# MODEL_TYPE = "NNdzdt"
# MODEL_TYPE = "AE-SINDy"

if MODEL_TYPE == "AE-AR":
    from training_scripts.train_ae_ar import AEAutoregressor, params, train_and_eval
elif MODEL_TYPE == "NNdzdt":
    from training_scripts.train_ae_NNdzdt import AENNdzdt, params, train_and_eval
elif MODEL_TYPE == "AE-SINDy":
    from training_scripts.train_ae_sindy import AESINDy, params, train_and_eval
    from src import thresholding
else:
    raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")


if __name__ == "__main__":
    total_trials = 48  # On mac with 8 perf. cores, choose multiple of 8 total_trials
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
    base_output_directory = Path("./")
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
