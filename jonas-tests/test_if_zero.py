# check if the SINDy model coefficients are actual zero for mass
import sys
import torch
import numpy as np

sys.path.append("/g/g14/katona1/mphys-surrogate-model")

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
    path="/g/g14/katona1/mphys-surrogate-model/data/congestus_coal_200m",
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

main_path = "/g/g14/katona1/mphys-surrogate-model/results/poster_erf_results/"
ae_sindy_checkpoint = torch.load(
    main_path
    + "ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth",
    weights_only=True,
)

# %%
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(ae_sindy_checkpoint)

print(ae_sindy.dzdt.sindy_coeffs.weight)
