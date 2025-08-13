"""
Script takes in data filepath and does conformal predictions on AE-AR model.
"""
import os
import sys
import numpy as np
import argparse
import torch
import time
from pathlib import Path
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import training
from training_scripts import train_ae_ar as train

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t",
    "--test_size",
    type=float,
    default=0.2,
    help="testing set proportion, must be between 0 and 1, default is 0.2",
)
parser.add_argument(
    "-m",
    "--method",
    default="split30",
    help="conformal predictions method to use (split[p], full), default is split30. \
                        The p in split indicates what *percent* you want to dedicate out of the full data for calibration.",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="number of epochs (default is 100)"
)
parser.add_argument(
    "-b", "--batches", type=int, default=200, help="batch size (default is 200)"
)
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=0.1,
    help="miscoverage rate(s), must be between 0 and 1, default is 0.1",
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
    "num_epochs": args.epochs,
    "batch_size": args.batches,
    "learning_rate": 0.002482884780966882,
    "latent_dim": 3,
    "n_lag": 1,
    "w_recon": 1,
    "w_dx": 0.22816989332325596,
    "w_dz": 0.6719555656053005,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "layer_size": (63, 98, 30),
    "CNN": False,
    "print_frequency": 1,
}

# Global variables and settings
# Criterion and divergence need to be outside train function to be available in other scripts
torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])
sample_time = None  # for setting times to sample, as indices
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

# Setting inputted variables
test_size = args.test_size
method = args.method
calib_size = None  # default is no calibration data
# isolate calib_size or k in the situation where you're using split-conformal or cv+, respectively
if method[:5] == "split":
    calib_size = float(method[5:])
    if (calib_size <= 0) | (calib_size >= 100 * (1 - test_size)):
        raise ValueError(
            "Calibration size for split must be a percent strictly between 0 and 100 * (1 - test_size)."
        )
    method = "split"
if method[:3] == "cv+":
    k = int(method[3:])
    method = "cv+"
if method not in [
    "split",
    "full",
    "cv+",
]:  # raise error if method is not one of the list above
    raise ValueError("Conformal predictions method specified has not been implemented.")

alphas = args.alpha if isinstance(args.alpha, (list, tuple)) else [args.alpha]

if (test_size <= 0) | (test_size >= 1):
    raise ValueError("Test set proportion must be between 0 and 1")
if any(a <= 0 or a >= 1 for a in alphas):
    raise ValueError("Coverage rate (alpha) must be between 0 and 1 for all values")

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()

# Set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available() and params["batch_size"] > 1000
        else "cpu"
    )
)
# torch.backends.cudnn.benchmark = True
print(f"Using {device} device")

if calib_size is not None:
    calib_size *= 0.01
start_time = time.time()
# Open dataset
outputs = du.open_mass_dataset(
    name=params["data_src"],
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=sample_time,
    test_size=test_size,
    calib_size=calib_size,
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
        CNN=params["CNN"],
    )
    optimal_path = os.path.join(
        "results",
        "Optuna",
        "ERF Dataset",
        "AE-AR_2025-07-20T22:45:33_605d8b8697694137a65cab3b1012fffc",
        "erf_FFNN_latent3_order(63, 98, 30)_tr1000_lr0.002482884780966882_bs4_weights0.22816989332325596-0.6719555656053005_7c43ff3e659b47358fa327f690871ed7",
    )
    ae_ar_checkpoint = torch.load(
        os.path.join(
            optimal_path,
            "erf_FFNN_latent3_order(63, 98, 30)_tr1000_lr0.002482884780966882_bs4_weights0.22816989332325596-0.6719555656053005_7c43ff3e659b47358fa327f690871ed7.pth",
        ),
        weights_only=True,
    )
    model.load_state_dict(ae_ar_checkpoint)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    return model.to(device)


# initialize optimizer
def init_optimizer(model):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    return optimizer


# initialize scheduling
def init_scheduler(optimizer):
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return sched


early_stopping = training.EarlyStopping(patience=params["patience"])

"""
Helper utility functions
"""


# this helper utility encodes and then decodes each DSD to test how close the predictions are to the actual values
def encode_decode_ae(x, model):
    # for DSD data, decoded (predictions from network)
    DSD_all = (
        model.decoder(model.encoder(torch.Tensor(x))).detach().numpy()
    )  # DSD pred t+0
    return DSD_all


# run latent space dynamics
def run_ae_X_latent(x, m, model, n_lag=params["n_lag"]):
    # for DSD_mass data, not decoded (so still within the latent space)
    latents_all = np.empty(
        (x.shape[0], x.shape[1], params["latent_dim"] + 1),
        dtype=float,
    )
    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        x0 = x[id, :n_lag, :]
        m0 = m[id, :n_lag]
        # encode DSD
        latents_all[id][:n_lag, :-1] = model.encoder(torch.Tensor(x0)).detach().numpy()
        latents_all[id][:n_lag, -1] = m0
        for t in range(n_lag, x.shape[1]):
            latent_DSD = torch.Tensor(
                latents_all[id][t - n_lag : t, :-1].reshape(
                    -1, n_lag, latents_all[id][t, :-1].shape[0]
                )
            )
            mass_t = torch.Tensor(
                latents_all[id][t - n_lag : t, -1].reshape(1, 1, 1)
            )  # reshape mass
            # run autoregressor in latent space
            res = model.autoregressor(torch.cat([latent_DSD, mass_t], dim=2))
            latents_all[id][t] = res.detach().numpy()
    return latents_all


# run full network
# computes the predicted DSD and mass trajectories
def run_ae_X(x, m, model, n_lag=params["n_lag"]):
    # DSD data, decoded (predictions from network)
    DSD_all = np.empty(x.shape, dtype=float)
    # mass data
    M_all = np.empty(m.shape, dtype=float)
    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        x0 = x[id, :n_lag, :]
        m0 = m[id, :n_lag]
        DSD_all[id][:n_lag, :] = (
            model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
        )
        M_all[id][:n_lag] = m0
        for t in range(n_lag, x.shape[1]):
            latent_DSD = model.encoder(
                torch.Tensor(
                    DSD_all[id][t - n_lag : t].reshape(
                        -1, n_lag, DSD_all[id][t].shape[0]
                    )
                )
            )  # encode DSD
            mass_t = torch.Tensor(
                M_all[id][t - n_lag : t].reshape(1, 1, 1)
            )  # reshape mass
            # run autoregressor in latent space
            res = model.autoregressor(torch.cat([latent_DSD, mass_t], dim=2))
            # decode the DSD and save it
            DSD_all[id][t, :] = model.decoder(res[..., :-1]).detach().numpy()
            M_all[id][t] = res[..., -1].squeeze().detach().item()  # save mass
    return DSD_all, M_all


# runs all three parts of the architecture above and returns data
def run_all(x, m, model, n_lag=params["n_lag"]):
    DSD_all = [
        np.empty(x.shape, dtype=float),  # decoder only
        np.empty(
            (x.shape[0], x.shape[1], params["latent_dim"]), dtype=float
        ),  # latent only
        np.empty(x.shape, dtype=float),  # full architecture
    ]
    M_all = np.empty(
        m.shape,
        dtype=float,
    )
    DSD_all[0] = encode_decode_ae(x, model=model)
    DSD_all[1] = run_ae_X_latent(x, m, model=model, n_lag=n_lag)[..., :-1]
    DSD_all[2], M_all = run_ae_X(x, m, model=model, n_lag=n_lag)
    return DSD_all, M_all


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


"""
Run conformal predictions. 
The data splits used are different for each method, so we will configure those separately in each case.
"""

if method == "full":
    # 0) configure data and dataloaders
    train_data = du.NormedBinDatasetAR(
        outputs["x_train"], outputs["m_train"], lag=params["n_lag"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_data = du.NormedBinDatasetAR(
        outputs["x_test"], outputs["m_test"], lag=params["n_lag"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # 1) fit & predict on training data
    DSD_lower_full = [
        np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),  # decoder only
        np.empty(
            (
                len(alphas),
                outputs["x_test"].shape[0],
                outputs["x_test"].shape[1],
                params["latent_dim"],
            ),
            dtype=float,
        ),  # latent only
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

    model = init_model(device)
    optimizer = init_optimizer(model)
    sched = init_scheduler(optimizer)

    (
        best_model,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = train.train_and_eval(
        params["num_epochs"],
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        params,
        early_stopping=early_stopping,
        print_flag=True,
        device=device,
    )
    print("Predicting the training data.")
    DSD_train_all, M_train_all = run_all(
        x=outputs["x_train"], m=outputs["m_train"], model=best_model
    )
    print("Predicting the testing data.")
    DSD_test_all, M_test_all = run_all(
        x=outputs["x_test"], m=outputs["m_test"], model=best_model
    )
    print("Running vanilla conformal predictions.")
    for i in range(3):  # loop through different subsets of the architecture
        if (
            i == 1
        ):  # in latent case, you should do it relative to latent space. Otherwise, not
            DSD_res_signed = (
                DSD_train_all[i]
                - best_model.encoder(torch.Tensor(outputs["x_train"])).detach().numpy()
            )
        else:
            DSD_res_signed = DSD_train_all[i] - outputs["x_train"]
        DSD_q_low, DSD_q_high = one_sided_quantiles(
            DSD_res_signed, alpha_lows, alpha_ups
        )
        DSD_lower_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
        )
        DSD_upper_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]
        )
        if i == 2:  # also check mass in full architecture case
            M_res_signed = M_train_all - outputs["m_train"]
            M_q_low, M_q_high = one_sided_quantiles(M_res_signed, alpha_lows, alpha_ups)
            M_lower_full = M_test_all[np.newaxis, ...] - M_q_high[:, np.newaxis, ...]
            M_upper_full = M_test_all[np.newaxis, ...] - M_q_low[:, np.newaxis, ...]
    lower = DSD_lower_full
    upper = DSD_upper_full
    rep_DSD = DSD_test_all
    lower_m = M_lower_full
    upper_m = M_upper_full
    rep_m = M_test_all

if method == "split":
    # 0) configure data and dataloaders
    train_data = du.NormedBinDatasetAR(
        outputs["x_train"], outputs["m_train"], lag=params["n_lag"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    calib_data = du.NormedBinDatasetAR(
        outputs["x_calib"], outputs["m_calib"], lag=params["n_lag"]
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=len(calib_data), shuffle=True
    )
    test_data = du.NormedBinDatasetAR(
        outputs["x_test"], outputs["m_test"], lag=params["n_lag"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # 1) fit & predict on training data
    DSD_lower_full = [
        np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),  # decoder only
        np.empty(
            (
                len(alphas),
                outputs["x_test"].shape[0],
                outputs["x_test"].shape[1],
                params["latent_dim"],
            ),
            dtype=float,
        ),  # latent only
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

    model = init_model(device)
    optimizer = init_optimizer(model)
    sched = init_scheduler(optimizer)

    (
        best_model,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = train.train_and_eval(
        params["num_epochs"],
        model,
        train_loader,
        test_loader,
        optimizer,
        sched,
        params,
        early_stopping=early_stopping,
        print_flag=True,
        device=device,
    )
    print("Predicting the calibration data.")
    DSD_calib_all, M_calib_all = run_all(
        x=outputs["x_calib"], m=outputs["m_calib"], model=best_model
    )
    print("Predicting the testing data.")
    DSD_test_all, M_test_all = run_all(
        x=outputs["x_test"], m=outputs["m_test"], model=best_model
    )
    print("Running split conformal predictions.")
    for i in range(3):  # loop through different subsets of the architecture
        if (
            i == 1
        ):  # in latent case, you should do it relative to latent space. Otherwise, not
            DSD_res_signed = (
                DSD_calib_all[i]
                - best_model.encoder(torch.Tensor(outputs["x_calib"])).detach().numpy()
            )
        else:
            DSD_res_signed = DSD_calib_all[i] - outputs["x_calib"]
        DSD_q_low, DSD_q_high = one_sided_quantiles(
            DSD_res_signed, alpha_lows, alpha_ups
        )
        DSD_lower_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_high[:, np.newaxis, ...]
        )
        DSD_upper_full[i] = (
            DSD_test_all[i][np.newaxis, ...] - DSD_q_low[:, np.newaxis, ...]
        )
        if i == 2:  # also check mass in full architecture case
            M_res_signed = M_calib_all - outputs["m_calib"]
            M_q_low, M_q_high = one_sided_quantiles(M_res_signed, alpha_lows, alpha_ups)
            M_lower_full = M_test_all[np.newaxis, ...] - M_q_high[:, np.newaxis, ...]
            M_upper_full = M_test_all[np.newaxis, ...] - M_q_low[:, np.newaxis, ...]
    lower = DSD_lower_full
    upper = DSD_upper_full
    rep_DSD = DSD_test_all
    lower_m = M_lower_full
    upper_m = M_upper_full
    rep_m = M_test_all

"""
Save:
-alpha values,
-indices for test data, 
-(lower and upper interval DSD values, and representative ("center") DSDs), and
-(lower and upper interval mass values, and representative ("center") mass),
in that order.
"""

if method == "split":  # add split percent if needed
    method += str(int(100 * calib_size))
with open(
    os.path.join(
        "UQ", "conformal", "results", "ae_AR", args.data_name + "_" + method + ".pkl"
    ),
    "wb",
) as f:
    pickle.dump(
        [
            alphas,
            test_size,
            outputs["idx_test"],
            (lower, upper, rep_DSD),
            (lower_m, upper_m, rep_m),
        ],
        f,
    )
