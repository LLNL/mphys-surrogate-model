"""
Script takes in data filepath and does split conformal predictions on AE-AR model.
Assumes an already pretrained model and only does calibration and prediction on the specified data.
"""
import os
import sys
import numpy as np
import argparse
import torch

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
import time
from pathlib import Path
import pickle
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from training_scripts import train_ae_ar as train
from math import ceil
from sklearn.covariance import LedoitWolf

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-p",
    "--p",
    type=int,
    default=20,
    help="The p indicates what *percent* you want to dedicate out of the full data for calibration."
    "Default is 20%, in which case p=20.",
)
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=0.1,
    help="miscoverage rate(s), must be between 0 and 1, default is 0.1",
)
parser.add_argument(
    "-j",
    "--n_jobs",
    type=int,
    default=-1,
    help="The number of jobs allocatable for parallelizing the covariance computation."
    "Default is -1 (all).",
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
    "latent_dim": 3,
    "n_lag": 1,
    "layer_size": [141, 154, 40],
}

# Global variables and settings
# Criterion and divergence need to be outside train function to be available in other scripts
torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
sample_time = None  # for setting times to sample, as indices
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

# Setting inputted variables
calib_size = args.p

alphas = args.alpha if isinstance(args.alpha, (list, tuple)) else [args.alpha]

if any(a <= 0 or a >= 1 for a in alphas):
    raise ValueError("Coverage rate (alpha) must be between 0 and 1 for all values")

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
print(f"Using {device} device")

calib_size *= 0.01
start_time = time.time()
# Open dataset
outputs = du.open_mass_dataset(
    name=params["data_src"],
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=sample_time,
    test_size=1 - calib_size,
    random_state=params["random_seed"],
)


# Initialize the model using optimal weights from results/Optuna
def init_model(device=device):
    model = train.AEAutoregressor(
        n_channels=1,
        n_bins=outputs["n_bins"],
        n_latent=params["latent_dim"],
        n_lag=params["n_lag"],
        layer_size=params["layer_size"],
    )
    optimal_path = os.path.join(
        "results",
        "Optuna",
        "ERF Dataset",
        "AE-AR_2025-09-18T10:18:57_PostARBugfix",
        "erf_FFNN_latent3_order(141, 154, 40)_tr1000_lr0.0030348411572892766_bs8_weights0.12789450188986579-1.1729901013704414_a0f49326688d4e69bcc0e9a78da3c870",
    )
    ae_ar_checkpoint = torch.load(
        os.path.join(
            optimal_path,
            "erf_FFNN_latent3_order(141, 154, 40)_tr1000_lr0.0030348411572892766_bs8_weights0.12789450188986579-1.1729901013704414_a0f49326688d4e69bcc0e9a78da3c870.pth",
        ),
        weights_only=True,
    )
    model.load_state_dict(ae_ar_checkpoint)
    return model.to(device, dtype=torch.float32)


"""
Helper utility functions
"""


# this helper utility encodes and then decodes each DSD to test how close the predictions are to the actual values
def encode_decode_ae_batched(x_t, model):
    with torch.inference_mode():
        return model.decoder(model.encoder(x_t)).detach().cpu().numpy()


# run latent space dynamics
def run_ae_X_latent_batched(z_init_enc_t, m_t, model, n_lag):
    """
    z_init_enc_t: (N, T, L) torch on device (encoder(x))
    m_t:          (N, T)     torch on device
    Returns:
        latents_all_t: (N, T, L+1) torch on device
    """
    N, T, L = z_init_enc_t.shape
    latents_all = torch.empty(
        (N, T, L + 1), device=z_init_enc_t.device, dtype=z_init_enc_t.dtype
    )

    # seed with first n_lag frames (just copy encoded z + m)
    latents_all[:, :n_lag, :L] = z_init_enc_t[:, :n_lag, :]
    latents_all[:, :n_lag, L] = m_t[:, :n_lag]

    # roll forward autoregressively, batched over N
    for t in range(n_lag, T):
        # latents_all: (N, T, L+1); split into z (L) and mass (1)
        z_hist = latents_all[:, t - n_lag : t, :L]  # (N, n_lag, L)
        z_hist_f = z_hist.reshape(z_hist.shape[0], 1, -1)  # (N, 1, n_lag*L)

        # Use the most recent mass (t-1) to match training forward()
        m_last = latents_all[:, t - 1, L].unsqueeze(1).unsqueeze(2)  # (N, 1, 1)

        ar_in = torch.cat([z_hist_f, m_last], dim=2)  # (N, 1, n_lag*L + 1)

        res = model.autoregressor(ar_in)  # (N, 1, L+1)
        res = res.squeeze(1)  # (N, L+1)

        latents_all[:, t, :] = res

    return latents_all  # torch


# run full network
# computes the predicted DSD and mass trajectories
def run_ae_X_batched(x_t, m_t, model, n_lag):
    """
    x_t: (N, T, n_bins) torch on device
    m_t: (N, T)         torch on device
    Returns:
        DSD_all: (N, T, n_bins) numpy
        M_all:   (N, T)         numpy
        Z_all:   (N, T, L)      numpy
    """
    with torch.inference_mode():
        # encode once
        z_enc_t = model.encoder(x_t)  # (N, T, L)
        # run latent AR for whole batch
        latents_all = run_ae_X_latent_batched(z_enc_t, m_t, model, n_lag)  # (N, T, L+1)

        # decode all timesteps in one go
        z_only = latents_all[..., :-1].reshape(-1, z_enc_t.shape[-1])  # (N*T, L)
        DSD_flat = model.decoder(z_only)  # (N*T, n_bins)
        DSD_all_t = DSD_flat.reshape(x_t.shape[0], x_t.shape[1], -1)  # (N, T, n_bins)

        M_all_t = latents_all[..., -1]  # (N, T)

    # move back to cpu only once
    return (
        DSD_all_t.detach().cpu().numpy(),
        M_all_t.detach().cpu().numpy(),
        latents_all[..., :-1].detach().cpu().numpy(),
    )


# runs all three parts of the architecture above and returns data
def run_all_batched(outputs, model, n_lag, device):
    with torch.inference_mode():
        x_train_t = torch.tensor(outputs["x_train"], device=device, dtype=torch.float32)
        x_test_t = torch.tensor(outputs["x_test"], device=device, dtype=torch.float32)
        m_train_t = torch.tensor(outputs["m_train"], device=device, dtype=torch.float32)
        m_test_t = torch.tensor(outputs["m_test"], device=device, dtype=torch.float32)

    DSD_train_dec = encode_decode_ae_batched(x_train_t, model)
    DSD_test_dec = encode_decode_ae_batched(x_test_t, model)

    DSD_train_full, M_train_full, Z_train_full = run_ae_X_batched(
        x_train_t, m_train_t, model, n_lag
    )
    DSD_test_full, M_test_full, Z_test_full = run_ae_X_batched(
        x_test_t, m_test_t, model, n_lag
    )

    DSD_all_train = [DSD_train_dec, DSD_train_full, Z_train_full]
    DSD_all_test = [DSD_test_dec, DSD_test_full, Z_test_full]
    return (DSD_all_train, M_train_full), (DSD_all_test, M_test_full)


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

    # 1) build the full list of levels
    lows = np.array(alpha_lows)
    ups = 1.0 - np.array(alpha_ups)
    all_q = np.concatenate([lows, ups])  # e.g. [0.125, 0.025, 0.875, 0.975]

    # 2) sort levels and remember how to invert
    sort_idx = np.argsort(all_q)
    q_sorted = all_q[sort_idx]

    # 3) single quantile call
    qs = np.quantile(residuals, q_sorted, axis=0)

    # 4) invert the sort
    qs_unsorted = np.empty_like(qs)
    qs_unsorted[sort_idx] = qs

    # 5) split into lows / highs
    m = len(alpha_lows)
    q_low = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high


# for computing Mahalanobis distance scoring and covariance matrix inverses
def compute_scores_and_inv(res_t):
    # res_t: shape (m, d) at a fixed time t
    mu_t = res_t.mean(axis=0, keepdims=True)  # (1, d)
    res_c = res_t - mu_t  # center residuals
    lw = LedoitWolf().fit(res_c)  # covariance of centered residuals
    Sigma_inv = lw.precision_
    scores_t = np.einsum("mi,ij,mi->m", res_c, Sigma_inv, res_c)
    return scores_t, Sigma_inv, mu_t.squeeze(0)


"""
Run split conformal predictions. 
"""

# 1) predict on calibration and test data
DSD_lower_full = [
    np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),  # decoder only
    np.empty(
        (len(alphas),) + outputs["x_test"].shape, dtype=float
    ),  # full architecture
]
DSD_upper_full = DSD_lower_full.copy()
M_lower_full = np.empty(
    (len(alphas),) + outputs["m_test"].shape,
    dtype=float,
)
M_upper_full = M_lower_full.copy()

print("Loading and initializing model.")
model = init_model(device)
model.eval()

print("Encoding data.")
with torch.inference_mode():
    x_calib_t = torch.tensor(outputs["x_train"], device=device, dtype=torch.float32)
    x_test_t = torch.tensor(outputs["x_test"], device=device, dtype=torch.float32)
    z_calib_enc_t = model.encoder(x_calib_t)  # (N_calib, T, latent_dim)
    z_test_enc_t = model.encoder(x_test_t)  # (N_test,  T, latent_dim)

z_calib_enc = z_calib_enc_t.detach().cpu().numpy()
z_test_enc = z_test_enc_t.detach().cpu().numpy()

print("Predicting the calibration and testing data.")
(DSD_calib_all, M_calib_all), (DSD_test_all, M_test_all) = run_all_batched(
    outputs, model, params["n_lag"], device
)
print("Running split conformal predictions on decoder and full outputs.")
for i in range(2):  # loop through decoder and then full subsets of architecture
    DSD_res_signed = DSD_calib_all[i] - outputs["x_train"]
    DSD_q_low, DSD_q_high = one_sided_quantiles(DSD_res_signed, alpha_lows, alpha_ups)
    DSD_lower_full[i] = (
        DSD_test_all[i][np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
    )
    DSD_upper_full[i] = DSD_test_all[i][np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]
    if i == 1:  # also check mass in full architecture case
        M_res_signed = M_calib_all - outputs["m_train"]
        M_q_low, M_q_high = one_sided_quantiles(M_res_signed, alpha_lows, alpha_ups)
        M_lower_full = M_test_all[np.newaxis, ...] - M_q_high[:, np.newaxis, ...]
        M_upper_full = M_test_all[np.newaxis, ...] - M_q_low[:, np.newaxis, ...]
# conformal predictions on latent trajectories
print("Running split conformal predictions on latent trajectories.")
residuals_lat = DSD_calib_all[2] - z_calib_enc  # (m, n, d)

m, n, d = residuals_lat.shape
results = Parallel(n_jobs=args.n_jobs)(
    delayed(compute_scores_and_inv)(residuals_lat[:, t, :]) for t in range(n)
)

scores_list, Sigma_inv_list, mu_list = zip(*results)
scores = np.stack(scores_list, axis=1)  # (m, n)
Sigma_inv = np.stack(Sigma_inv_list, axis=0)  # (n, d, d)
mu = np.stack(mu_list, axis=0)  # (n, d)

taus = np.zeros((len(alphas), scores.shape[1]))
for i, alpha in enumerate(alphas):
    k = min(ceil((len(scores) + 1) * (1 - alpha)), len(scores))
    taus[i] = np.sort(scores, axis=0)[k - 1]

latent_dict = {
    "z_enc_test_pred": DSD_test_all[2],
    "Sigma_inv": Sigma_inv,
    "mu": mu,  # <-- save the mean!
    "taus": taus,
}
lower = DSD_lower_full
upper = DSD_upper_full
rep_DSD = DSD_test_all
lower_m = M_lower_full
upper_m = M_upper_full
rep_m = M_test_all

print("Saving results.")

"""
Save:
-alpha values,
-indices for test data, 
-(lower and upper interval DSD values, and representative ("center") DSDs), 
-(lower and upper interval mass values, and representative ("center") mass), and
-{encoded predictions on test set, covariance inverse of latent space trajectories, residual mean, conformal radii}
in that order.
"""

with open(
    os.path.join(
        "UQ",
        "conformal",
        "results",
        "ae_AR",
        os.path.basename(os.path.normpath(args.data_name))
        + "_split"
        + str(int(100 * calib_size))
        + ".pkl",
    ),
    "wb",
) as f:
    pickle.dump(
        [
            alphas,
            outputs["idx_test"],
            (lower, upper, rep_DSD),
            (lower_m, upper_m, rep_m),
            latent_dict,
        ],
        f,
    )
