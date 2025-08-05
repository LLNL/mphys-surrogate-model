"""
Script takes in data filepath and does conformal predictions on AE-NNdzdt model.
"""
import os
import sys
import numpy as np
import argparse
import torch
import time
from pathlib import Path
from contextlib import redirect_stdout

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import training
from training_scripts import train_ae_NNdzdt as train

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
    default="cv+5",
    help="conformal predictions method to use (jackknife, split[p], full, cv+[k]), default is cv+5. \
                        The p in split indicates what *percent* you want to dedicate out of the full data for calibration, while\
                            the k in cv+[k] indicates how many models to train for cross-validation folds.",
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
    "num_epochs": 500,
    "batch_size": 100,
    "learning_rate": 0.00314227212817401,
    "latent_dim": 3,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "layer_size": (42, 36, 46),
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

# Initialize the model
model = train.AENNdzdt(
    n_channels=1,
    n_bins=outputs["n_bins"],
    n_latent=params["latent_dim"],
    layer_size=params["layer_size"],
    CNN=params["CNN"],
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

# Optimizer and scheduling
optimizer = torch.optim.AdamW(
    model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
early_stopping = training.EarlyStopping(patience=params["patience"])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Compute & set weights based on Champion et al recs
lambda1, lambda2, lambda3 = du.champion_calculate_weights(
    du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    ),
    lambda1_metaweight=0.5353139650038768,
)
print(f"lambda: 1.0, {lambda1}, {lambda2}")
params["loss_weight_recon"] = 1.0
params["loss_weight_sindy_x"] = lambda1
params["loss_weight_sindy_z"] = lambda2

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
def run_ae_X_latent(x, m, model):
    # for DSD_mass data, not decoded (so still within the latent space)
    latents_all = np.empty(
        (x.shape[0], x.shape[1], params["latent_dim"] + 1),
        dtype=float,
    )
    z_enc_train = model.encoder(torch.Tensor(x)).detach().numpy()
    zlim = np.zeros((3 + 1, 2))  # DSD bins
    for il in range(3):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m.min()
    zlim[-1][1] = m.max()

    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        z0 = np.concatenate((z_enc_train[id, 0, :], np.array([m[id, 0]])), axis=-1)
        latents_all[id] = du.simulate(z0, outputs["dsd_time"], model.dzdt, zlim)
    return latents_all


# run full network
# computes the predicted DSD and mass trajectories
def run_ae_X(x, m, model):
    # DSD data, decoded (predictions from network)
    DSD_all = np.empty(x.shape, dtype=float)
    # mass data
    M_all = np.empty(m.shape, dtype=float)
    z_enc_train = model.encoder(torch.Tensor(x)).detach().numpy()
    zlim = np.zeros((3 + 1, 2))  # DSD bins
    for il in range(3):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m.min()
    zlim[-1][1] = m.max()

    for id in range(len(x)):  # loop through each initial condition/sample/gridbox
        z0 = np.concatenate((z_enc_train[id, 0, :], np.array([m[id, 0]])), axis=-1)
        latents_pred = du.simulate(z0, outputs["dsd_time"], model.dzdt, zlim)
        DSD_all[id] = (
            model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()
        )  # add DSD predictions
        M_all[id] = latents_pred[:, -1]  # add mass predictions
    return DSD_all, M_all


# runs all three parts of the architecture above and returns data
def run_all(x, m, model):
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
    DSD_all[1] = run_ae_X_latent(x, m, model=model)[..., :-1]
    DSD_all[2], M_all = run_ae_X(x, m, model=model)
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
    train_data = du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_data = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
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
    # suppress output of train_and_eval
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull):
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

if method == "split":
    # 0) configure data and dataloaders
    train_data = du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    calib_data = du.NormedBinDatasetDzDt(
        outputs["x_calib"], outputs["dsd_time"], outputs["m_calib"]
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_data, batch_size=len(calib_data), shuffle=True
    )
    test_data = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
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
    # suppress output of train_and_eval
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull):
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
    print("Running vanilla conformal predictions.")
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

if method == "cv+":
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=1952)

    # collect residuals for each fold and each architecture subset of shape (n_test, T, D)
    DSD_resid = [[], [], []]
    M_resid = []
    # collect per-fold predictions and each architecture subset of shape (n_test, T, D)
    DSD_pred = [[], [], []]
    M_pred = []

    # 0) configure test data and test dataloader
    test_data = du.NormedBinDatasetDzDt(
        outputs["x_test"], outputs["dsd_time"], outputs["m_test"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )
    print(f"Running {k}-fold cross validation.")
    for j, idx in enumerate(kf.split(outputs["x_train"])):
        train_idx = idx[0]
        val_idx = idx[1]
        # 0) configure data and dataloaders
        train_data = du.NormedBinDatasetDzDt(
            outputs["x_train"][train_idx],
            outputs["dsd_time"],
            outputs["m_train"][train_idx],
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=params["batch_size"], shuffle=True
        )
        calib_data = du.NormedBinDatasetDzDt(
            outputs["x_train"][val_idx],
            outputs["dsd_time"],
            outputs["m_train"][val_idx],
        )
        calib_loader = torch.utils.data.DataLoader(
            calib_data, batch_size=len(val_idx), shuffle=True
        )

        # 1) fit on fold‐k train
        # suppress output of train_and_eval
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull):
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
        # 2) predict on validation (calibration) fold
        DSD_calib_all, M_calib_all = run_all(
            x=outputs["x_train"][val_idx],
            m=outputs["m_train"][val_idx],
            model=best_model,
        )
        DSD_test_all, M_test_all = run_all(
            x=outputs["x_test"], m=outputs["m_test"], model=best_model
        )
        for i in range(3):  # loop through different subsets of the architecture
            # 3) signed residuals on calibration fold
            if (
                i == 1
            ):  # in latent case, you should do it relative to latent space. Otherwise, not
                DSD_resid[i].append(
                    DSD_calib_all[i]
                    - best_model.encoder(torch.Tensor(outputs["x_train"][val_idx]))
                    .detach()
                    .numpy()
                )
            else:
                DSD_resid[i].append(DSD_calib_all[i] - outputs["x_train"][val_idx])
            # 4) predict on the TEST set
            DSD_pred[i].append(DSD_test_all[i])
            if i == 2:  # also check mass in full architecture case
                M_resid.append(M_calib_all - outputs["m_train"][val_idx])
                M_pred.append(M_test_all)
        print(f"Finished with fold {j+1}.")
    rep_DSD = [[], [], []]
    for i in range(3):
        # 5) pool residuals and compute global quantiles
        resid_pool = np.concatenate(DSD_resid[i], axis=0)
        ql, qh = one_sided_quantiles(resid_pool, alpha_lows, alpha_ups)

        # 6) stack & intersect across folds
        rep_DSD[i] = np.median(
            np.stack(DSD_pred[i], axis=0), axis=0
        )  # intersect via median of K-fold predictions
        lower = rep_DSD[i][np.newaxis, ...] - qh[:, np.newaxis, ...]
        upper = rep_DSD[i][np.newaxis, ...] - ql[:, np.newaxis, ...]
    # for mass
    resid_pool = np.concatenate(M_resid, axis=0)
    ql, qh = one_sided_quantiles(resid_pool, alpha_lows, alpha_ups)

    # 6) stack & intersect across folds
    rep_traj = np.median(
        np.stack(M_pred, axis=0), axis=0
    )  # intersect via median of K-fold predictions
    lower = rep_traj[np.newaxis, ...] - qh[:, np.newaxis, ...]
    upper = rep_traj[np.newaxis, ...] - ql[:, np.newaxis, ...]
