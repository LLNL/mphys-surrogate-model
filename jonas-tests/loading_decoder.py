# testing vanilla and split conformal predictions on pretrained model, decoding only

import os
import sys
import pickle
import torch
import numpy as np

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)

from src import data_utils as du
from training_scripts.train_ae_ar import AEAutoregressor
from training_scripts.train_ae_sindy import AESINDy
from training_scripts.train_ae_NNdzdt import AENNdzdt

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
    sample_time=np.arange(0, 61, 5), path=parent_directory + "/data/congestus_coal_200m"
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
ae_ar_checkpoint = torch.load(
    main_path
    + "ae_ar/model/erf_FFNN_latent3_order(10, 20, 10)_tr100_lr0.001_bs128_weights1-1_f74ecd55b87a43e2bfde2bf9973fdf10.pth",
    weights_only=True,
)
ae_nndzdt_checkpoint = torch.load(
    main_path
    + "ae_nndzdt/model/FFNN_latent3_layers(40, 40, 40)_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_7850dd2260f34875baf55452f4311d60.pth",
    weights_only=True,
)
ae_sindy_checkpoint = torch.load(
    main_path
    + "ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth",
    weights_only=True,
)

# %%
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(ae_sindy_checkpoint)
# %%
ae_nndzdt = AENNdzdt(
    n_channels=1, n_bins=64, n_latent=3, layer_size=(40, 40, 40), CNN=False
)
ae_nndzdt.load_state_dict(ae_nndzdt_checkpoint)
# %%
ae_ar = AEAutoregressor(
    n_channels=1, n_bins=64, n_latent=3, n_lag=1, layer_size=(10, 20, 10), CNN=False
)
ae_ar.load_state_dict(ae_ar_checkpoint)

models = (ae_sindy, ae_nndzdt, ae_ar)
model_label = ["SINDy", "NN-driven", "AR"]


# this helper utility encodes and then decodes each DSD to test how close the predictions are to the actual values
def encode_decode_ae(x):
    # for DSD data, decoded (predictions from network)
    DSD_all = np.empty((len(model_label),) + x.shape, dtype=float)
    for k, model in enumerate(models):
        DSD_all[k] = (
            model.decoder(model.encoder(torch.Tensor(x))).detach().numpy()
        )  # DSD pred t+0
    return DSD_all


def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    """
    Compute all lower- and upper-tail quantiles in one shot.

    residuals : array_like, shape (N, T, D)
    alpha_lows: list of alpha/2 levels (e.g. [0.125, 0.025] for 75% & 95%)
    alpha_ups : same as alpha_lows
    Returns
    -------
    q_low, q_high : arrays of shape (len(alpha_lows), T, D)
    """
    import numpy as _np

    # 1) build the full list of levels
    lows = _np.array(alpha_lows)
    ups = 1.0 - _np.array(alpha_ups)
    all_q = _np.concatenate([lows, ups])  # e.g. [0.125, 0.025, 0.875, 0.975]

    # 2) sort levels and remember how to invert
    sort_idx = _np.argsort(all_q)
    q_sorted = all_q[sort_idx]

    # 3) single quantile call
    qs = _np.quantile(residuals, q_sorted, axis=0)

    # 4) invert the sort
    qs_unsorted = _np.empty_like(qs)
    qs_unsorted[sort_idx] = qs

    # 5) split into lows / highs
    m = len(alpha_lows)
    q_low = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high


alphas = [0.25, 0.1, 0.05, 0.01]

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()

# full
# loop through all models to generate
DSD_lower_full = np.empty(
    (
        len(models),
        len(alphas),
    )
    + x_test.shape,
    dtype=float,
)
DSD_upper_full = DSD_lower_full.copy()
print("Predicting the training data.")
DSD_train_all = encode_decode_ae(x_train)
print("Predicting the testing data.")
DSD_test = encode_decode_ae(x_test)
print("Running vanilla conformal predictions.")
for i in range(len(models)):
    DSD_res_signed = DSD_train_all[i] - x_train
    DSD_q_low, DSD_q_high = one_sided_quantiles(DSD_res_signed, alpha_lows, alpha_ups)
    DSD_lower_full[i] = DSD_test[i, np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
    DSD_upper_full[i] = DSD_test[i, np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]

# save lower, upper, and representative for DSD
DSD_bands_full = [DSD_lower_full, DSD_test, DSD_upper_full]

# split
print("Splitting testing data into calibration and testing data. 50-50 split.")
# make a global index array
n_all = len(x_test)
idx_all = np.arange(n_all)  # [0,1,...,n_all-1]

from sklearn.model_selection import train_test_split

# calibration/test split on indices
idx_cal, idx_testing = train_test_split(idx_all, test_size=0.5, random_state=1952)

# split data into training and testing data (training includes calibration, FYI)
x_testing = x_test.copy()
x_cal = x_testing[idx_cal]
x_testing = x_testing[idx_testing]
DSD_lower_split = np.empty(
    (
        len(models),
        len(alphas),
    )
    + x_testing.shape,
    dtype=float,
)
DSD_upper_split = DSD_lower_split.copy()
print("Predicting the calibration data.")
DSD_cal = encode_decode_ae(x_cal)
print("Predicting the testing data.")
DSD_testing = encode_decode_ae(x_testing)
print("Running split conformal predictions.")
for i in range(len(models)):
    DSD_res_signed = DSD_cal[i] - x_cal
    DSD_q_low, DSD_q_high = one_sided_quantiles(DSD_res_signed, alpha_lows, alpha_ups)
    DSD_lower_split[i] = (
        DSD_testing[i, np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
    )
    DSD_upper_split[i] = DSD_testing[i, np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]

# save lower, upper, and representative for DSD
DSD_bands_split = [DSD_lower_split, DSD_testing, DSD_upper_split]

print("Pickling results.")

path = parent_directory + "/results/jonas-cp-tests"

# full/vanilla: save alpha values and prediction bands for DSDs (in that order)

with open(path + "/decoder/vanilla.pkl", "wb") as f:
    pickle.dump([alphas, DSD_bands_full], f)

# split: save alpha values, new testing indices (after test indices split into calibration+testing), and prediction bands for DSDs (in that order)

with open(path + "/decoder/split.pkl", "wb") as f:
    pickle.dump([alphas, idx_testing, DSD_bands_split], f)
