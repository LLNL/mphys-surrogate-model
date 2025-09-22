"""
Script takes in data filepath and does split conformal predictions on AE-NNdzdt model.
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
from training_scripts import train_ae_NNdzdt as train
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
parser.add_argument(
    "-ode",
    "--ODE",
    type=str,
    default="damped",
    choices=["damped", "clamped", "none"],
    help="Which safeguard to implement for the latent ODE solver."
    "Damped adds a polynomial damping term to each vector field to drive solutions that stray too far back to the origin."
    "Clamped limits the ODE solution within the bounds from the training data."
    "None implements no such safeguards (highly not recommended)."
    "Default is damped (add polynomial damping term to drive ODE solutions back to the origin).",
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
    "latent_dim": 3,
    "layer_size": (42, 36, 46),
    "eps": 1e-5,  # damping coefficient for ODE integration
    "p": 3,  # must be odd; damping exponent
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
    model = train.AENNdzdt(
        n_channels=1,
        n_bins=outputs["n_bins"],
        n_latent=params["latent_dim"],
        layer_size=params["layer_size"],
    )
    optimal_path = os.path.join(
        "results",
        "Optuna",
        "ERF Dataset",
        "NNdzdt_2025-07-20T23:31:20_3a400c596947422389559813cd41dfe6",
        "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3",
    )
    ae_NNdzdt_checkpoint = torch.load(
        os.path.join(
            optimal_path,
            "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3.pth",
        ),
        weights_only=True,
    )
    model.load_state_dict(ae_NNdzdt_checkpoint)
    return model.to(device, dtype=torch.float32)


"""
Helper utility functions
"""


# simulate multiple latent space initial conditions at once in parallel
def simulate_many(z0s, t_grid, dzdt, opt, n_jobs):
    if args.ODE == "damped":
        sims = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(du.simulate_damped)(z0, t_grid, dzdt, opt, params["p"])
            for z0 in z0s
        )
    else:
        sims = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(du.simulate)(z0, t_grid, dzdt, opt) for z0 in z0s
        )
    return np.stack(sims, axis=0)  # (N, T, latent_dim+1)


# this helper utility encodes and then decodes each DSD to test how close the predictions are to the actual values
def encode_decode_ae_batched(x_t, model):
    with torch.inference_mode():
        decoded = model.decoder(model.encoder(x_t))
    return decoded.detach().cpu().numpy()


# run latent space dynamics
def run_ae_X_latent_batched(z_enc, m, model, opt, n_jobs):
    # z_enc: (N, T, latent_dim) as numpy
    z0s = [
        np.concatenate((z_enc[i, 0, :], np.array([m[i, 0]])), axis=-1)
        for i in range(z_enc.shape[0])
    ]
    latents_all = simulate_many(
        z0s, outputs["dsd_time"], model.dzdt, opt, n_jobs
    )  # (N, T, L+1)
    return latents_all  # keep as numpy


# run full network
# computes the predicted DSD and mass trajectories
def run_ae_X_batched(z_enc, m, model, opt, n_jobs, device):
    latents_all = run_ae_X_latent_batched(z_enc, m, model, opt, n_jobs)  # (N, T, L+1)
    # Decode all DSDs in batches on device
    latents_only = latents_all[..., :-1]  # (N, T, L)
    N, T, L = latents_only.shape
    flat = latents_only.reshape(-1, L)
    with torch.inference_mode():
        flat_t = torch.tensor(flat, device=device, dtype=torch.float32)
        decoded_flat = model.decoder(flat_t)  # (N*T, n_bins)
        decoded = (
            decoded_flat.detach().cpu().numpy().reshape(N, T, -1)
        )  # (N, T, n_bins)
    M_all = latents_all[..., -1]  # (N, T)
    return decoded, M_all, latents_only


# runs all three parts of the architecture above and returns data
def run_all_batched(outputs, model, z_train_enc, z_test_enc, opt, device, n_jobs):
    # decoder-only on train/test
    DSD_train_dec_only = encode_decode_ae_batched(
        torch.tensor(outputs["x_train"], device=device, dtype=torch.float32), model
    )
    DSD_test_dec_only = encode_decode_ae_batched(
        torch.tensor(outputs["x_test"], device=device, dtype=torch.float32), model
    )

    # full architecture (simulate + decode) on train/test
    DSD_train_full, M_train_full, Z_train_full = run_ae_X_batched(
        z_train_enc, outputs["m_train"], model, opt, n_jobs, device
    )
    DSD_test_full, M_test_full, Z_test_full = run_ae_X_batched(
        z_test_enc, outputs["m_test"], model, opt, n_jobs, device
    )

    # package to match your original layout
    DSD_all_train = [DSD_train_dec_only, DSD_train_full, Z_train_full]
    DSD_all_test = [DSD_test_dec_only, DSD_test_full, Z_test_full]
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
Run conformal predictions. 
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
if args.ODE == "damped":
    (DSD_calib_all, M_calib_all), (DSD_test_all, M_test_all) = run_all_batched(
        outputs,
        model,
        z_calib_enc,
        z_test_enc,
        params["eps"],
        device,
        n_jobs=args.n_jobs,
    )
if args.ODE == "clamped":
    import xarray as xr

    ds = xr.open_dataset(
        (
            Path(__file__).parent.parent.parent / "data" / "congestus_coal_200m_train"
        ).with_suffix(".nc")
    )
    # put into target order once so we don't keep transposing
    dmdlnr = ds["dmdlnr"].transpose("loc", "t", "bin")

    # mass (loc, t), normalized mass, and normalized spectra (loc, t, bin)
    m = dmdlnr.sum(dim="bin")
    m_scale = m.max().item()

    x_train = (dmdlnr / m).to_numpy()
    m_train = (m / m_scale).to_numpy()

    with torch.inference_mode():
        z_train_enc = (
            model.encoder(torch.tensor(x_train, device=device, dtype=torch.float32))
            .detach()
            .cpu()
            .numpy()
        )

    zlim = np.zeros((params["latent_dim"] + 1, 2), dtype=float)
    for il in range(params["latent_dim"]):
        zlim[il, 0] = z_train_enc[:, :, il].min()
        zlim[il, 1] = z_train_enc[:, :, il].max()
    zlim[-1, 0] = m_train.min()
    zlim[-1, 1] = m_train.max()
    (DSD_calib_all, M_calib_all), (DSD_test_all, M_test_all) = run_all_batched(
        outputs, model, z_calib_enc, z_test_enc, zlim, device, n_jobs=args.n_jobs
    )
if args.ODE == "none":
    zlim = np.array([[-np.inf, np.inf]] * (params["latent_dim"] + 1), dtype=float)
    (DSD_calib_all, M_calib_all), (DSD_test_all, M_test_all) = run_all_batched(
        outputs, model, z_calib_enc, z_test_enc, zlim, device, n_jobs=args.n_jobs
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
        "ae_NNdzdt",
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
