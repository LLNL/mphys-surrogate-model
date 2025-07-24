# check if the SINDy model coefficients are actual zero for mass
import os
import sys
import numpy as np
import torch

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)

from src import data_utils as du
from training_scripts.train_ae_sindy import AESINDy

# %%
(
    x_train,  # (#samples, #timesteps, #dims)
    m_train,  # (#samples, #timesteps)
    x_test,
    m_test,
    r_bins_edges,
    n_bins,
    dsd_time,
) = du.open_erf_dataset(
    sample_time=np.arange(0, 61, 5),
    path=parent_directory + "/data/congestus_coal_200m",
)

params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "n_lag": 1,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    # "lambda3_sparsity": 0.0, TODO: sequential thresholding
    "CNN": False,
    "print_frequency": 1,
}

main_path = parent_directory + "/results/poster_erf_results/"
ae_sindy_checkpoint = torch.load(
    main_path
    + "ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth",
    weights_only=True,
)

# %%
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(ae_sindy_checkpoint)

print(ae_sindy.dzdt.sindy_coeffs.weight)
