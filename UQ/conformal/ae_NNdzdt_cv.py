#!/usr/bin/env python3
"""
Script: ae-NNdzdt_conformal_cvplus.py

Performs cv+ conformal prediction for the AE-NNdzdt model.
Only the cv+ (cross-validated) branch is supported.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch

# Seeds & threading controls
np.random.seed(1952)
torch.manual_seed(1952)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

from pathlib import Path
from multiprocessing import get_context
from sklearn.model_selection import KFold
from contextlib import redirect_stdout

# -- Make sure project root is on PYTHONPATH -----------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import training
from training_scripts import train_ae_NNdzdt as train

# ----------------------------------------------------------------------------
# Command-line arguments
parser = argparse.ArgumentParser(description="cv+ conformal for AE-NNdzdt")
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t", "--test_size", type=float, default=0.2, help="test split proportion [0,1]"
)
parser.add_argument("-k", "--folds", type=int, default=5, help="number of CV folds")
parser.add_argument(
    "-c", "--cpus", type=int, default=None, help="override number of CPUs to use"
)
parser.add_argument("-e", "--epochs", type=int, default=100, help="training epochs")
parser.add_argument("-b", "--batches", type=int, default=200, help="batch size")
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=[0.1],
    help="miscoverage rate(s); list of values in (0,1)",
)
args = parser.parse_args()

# ----------------------------------------------------------------------------
# Parameters dictionary
params = dict(
    data_src=args.data_name,
    random_seed=1952,
    num_epochs=args.epochs,
    batch_size=args.batches,
    learning_rate=0.00314227212817401,
    latent_dim=3,
    lr_sched=True,
    patience=50,
    tol=1e-8,
    wd=1e-3,
    layer_size=(42, 36, 46),
    CNN=False,
    print_frequency=1,
    folds=args.folds,
    alpha_lows=[a / 2 for a in args.alpha],
    alpha_ups=[a / 2 for a in args.alpha],
    optuna_tag="NNdzdt_2025-07-20T23:31:20_3a400c596947422389559813cd41dfe6",
    optuna_modelname=(
        "erf_FFNN_latent3_layers(42, 36, 46)_"
        "tr1000_lr0.00314227212817401_bs4_weights1.0-"
        "599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3/"
        "erf_FFNN_latent3_layers(42, 36, 46)_"
        "tr1000_lr0.00314227212817401_bs4_weights1.0-"
        "599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3"
    ),
)

# ----------------------------------------------------------------------------
# Load dataset
outputs = du.open_mass_dataset(
    name=args.data_name,
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=None,
    test_size=args.test_size,
    calib_size=None,
    random_state=1952,
)

# Compute & set weights based on Champion et al recs
lambda1, lambda2, _ = du.champion_calculate_weights(
    du.NormedBinDatasetDzDt(
        outputs["x_train"], outputs["dsd_time"], outputs["m_train"]
    ),
    lambda1_metaweight=0.5353139650038768,
)
params["loss_weight_recon"] = 1.0
params["loss_weight_sindy_x"] = lambda1
params["loss_weight_sindy_z"] = lambda2

# Validate alpha
if any(a <= 0 or a >= 1 for a in args.alpha):
    raise ValueError("Each alpha must lie in (0,1)")


def init_model(outputs, params, device):
    """Initialize AE-NNdzdt with optimal weights."""
    model = train.AENNdzdt(
        n_channels=1,
        n_bins=outputs["n_bins"],
        n_latent=params["latent_dim"],
        layer_size=params["layer_size"],
        CNN=params["CNN"],
    )
    optimal_path = Path("results") / "Optuna" / "ERF Dataset" / params["optuna_tag"]
    checkpoint = torch.load(
        optimal_path / f"{params['optuna_modelname']}.pth",
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    return model.to(device)


def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    """
    Compute lower- and upper-tail quantiles for multiple alphas in one pass.

    residuals : array_like, shape (N, T, D)
    alpha_lows: list of alpha/2 levels
    alpha_ups : same as alpha_lows
    Returns q_low, q_high arrays of shape (len(alpha_lows), T, D)
    """
    lows = np.array(alpha_lows)
    ups = 1.0 - np.array(alpha_ups)
    all_q = np.concatenate([lows, ups])
    sort_idx = np.argsort(all_q)
    q_sorted = all_q[sort_idx]
    qs = np.quantile(residuals, q_sorted, axis=0)
    qs_unsorted = np.empty_like(qs)
    qs_unsorted[sort_idx] = qs
    m = len(alpha_lows)
    return qs_unsorted[:m], qs_unsorted[m:]


def run_all(x, m, model, outputs, params):
    """
    Runs:
      0) encoder→decoder
      1) latent ODE only
      2) full simulate→decode
    Returns DSD_preds list of three arrays, plus M_preds
    """

    # helper subroutines
    def encode_decode(x_arr):
        return model.decoder(model.encoder(torch.Tensor(x_arr))).detach().numpy()

    def run_latent(x_arr, m_arr):
        # shape: (N, T, latent+1)
        N, T = x_arr.shape[:2]
        lat_all = np.empty((N, T, params["latent_dim"] + 1), dtype=float)
        z_enc = model.encoder(torch.Tensor(x_arr)).detach().numpy()
        # determine z-limits
        zlim = np.zeros((params["latent_dim"] + 1, 2))
        for d in range(params["latent_dim"]):
            zlim[d] = (z_enc[:, :, d].min(), z_enc[:, :, d].max())
        zlim[-1] = (m_arr.min(), m_arr.max())

        for i in range(N):
            z0 = np.concatenate([z_enc[i, 0, :], [m_arr[i, 0]]])
            lat_all[i] = du.simulate(z0, outputs["dsd_time"], model.dzdt, zlim)
        return lat_all

    def run_full(x_arr, m_arr):
        N, T = x_arr.shape[:2]
        DSD_pred = np.empty_like(x_arr, dtype=float)
        M_pred = np.empty_like(m_arr, dtype=float)
        lat_all = run_latent(x_arr, m_arr)
        DSD_pred = model.decoder(torch.Tensor(lat_all[:, :, :-1])).detach().numpy()
        M_pred = lat_all[:, :, -1]
        return DSD_pred, M_pred

    # run each piece
    D0 = encode_decode(x)
    L1 = run_latent(x, m)[..., :-1]
    D2, M2 = run_full(x, m)
    return [D0, L1, D2], M2


def _fold_worker(args):
    """
    Trains on train_idx, calibrates on val_idx, predicts on test set.
    Returns residuals and predictions for each sub-network.
    """
    (fold_id, train_idx, val_idx, outputs, params, device) = args

    # rebuild model & optimizer & scheduler
    model = init_model(outputs, params, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    early_stop = training.EarlyStopping(patience=params["patience"])

    # data splits
    xtr, dtr, mtr = (
        outputs["x_train"][train_idx],
        outputs["dsd_time"],
        outputs["m_train"][train_idx],
    )
    train_ds = du.NormedBinDatasetDzDt(xtr, dtr, mtr)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True
    )

    # we’ll predict on the held-out calibration and the fixed test
    xcal = outputs["x_train"][val_idx]
    mcal = outputs["m_train"][val_idx]
    calib_ds = du.NormedBinDatasetDzDt(xcal, dtr, mcal)
    calib_loader = torch.utils.data.DataLoader(
        calib_ds, batch_size=len(calib_ds), shuffle=False
    )

    test_ds = du.NormedBinDatasetDzDt(outputs["x_test"], dtr, outputs["m_test"])
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False
    )

    # train & early-stop
    with open(os.devnull, "w") as devnull:
        _ = train.train_and_eval(
            params["num_epochs"],
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            params,
            early_stopping=early_stop,
            print_flag=False,
            device=device,
        )

    # get raw preds
    D_calib, M_calib = run_all(xcal, mcal, model, outputs, params)
    D_test, M_test = run_all(
        outputs["x_test"], outputs["m_test"], model, outputs, params
    )

    # compute residuals per-subnetwork
    D_res, D_pred = [], []
    for i in range(3):
        if i == 1:
            # latent vs. encoder output
            enc_val = model.encoder(torch.Tensor(xcal)).detach().numpy()
            resid = D_calib[i] - enc_val
        else:
            resid = D_calib[i] - xcal
        D_res.append(resid)
        D_pred.append(D_test[i])

    # mass residuals
    M_res = M_calib - outputs["m_train"][val_idx]
    M_pred = M_test

    print(f"[Fold {fold_id+1}] done on PID {os.getpid()}")
    return D_res, D_pred, M_res, M_pred


def main():
    # ----------------------------------------------------------------------------
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------------
    # Determine CPU count
    if args.cpus is None:
        total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()
    else:
        total_cpus = args.cpus
    print(f"→ launching {args.folds}-fold evaluation on {total_cpus} {device} cores")

    # ----------------------------------------------------------------------------
    # Build fold splits
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=params["random_seed"])
    splits = list(kf.split(outputs["x_train"]))

    # ----------------------------------------------------------------------------
    # Launch parallel folds with spawn context
    fold_args = [
        (i, tr_idx, val_idx, outputs, params, device)
        for i, (tr_idx, val_idx) in enumerate(splits)
    ]

    ctx = get_context("spawn")
    with ctx.Pool(processes=total_cpus) as pool:
        results = pool.map(_fold_worker, fold_args)

    # ----------------------------------------------------------------------------
    # Aggregate residuals & predictions
    D_res_pool, D_pred_pool, M_res_list, M_pred_list = [], [], [], []
    # initialize
    for _ in range(3):
        D_res_pool.append([])
        D_pred_pool.append([])

    for Dres, Dpred, Mres, Mpred in results:
        for i in range(3):
            D_res_pool[i].append(Dres[i])
            D_pred_pool[i].append(Dpred[i])
        M_res_list.append(Mres)
        M_pred_list.append(Mpred)

    # compute global quantiles & medians
    lower_D, upper_D, rep_D = [], [], []
    alows, aups = params["alpha_lows"], params["alpha_ups"]
    for i in range(3):
        resid_cat = np.concatenate(D_res_pool[i], axis=0)
        ql, qh = one_sided_quantiles(resid_cat, alows, aups)
        pred_med = np.median(np.stack(D_pred_pool[i], axis=0), axis=0)
        lower_D.append(pred_med[np.newaxis, ...] - qh[:, np.newaxis, ...])
        upper_D.append(pred_med[np.newaxis, ...] - ql[:, np.newaxis, ...])
        rep_D.append(pred_med)

    # mass
    M_res_cat = np.concatenate(M_res_list, axis=0)
    ql_m, qh_m = one_sided_quantiles(M_res_cat, alows, aups)
    rep_m = np.median(np.stack(M_pred_list, axis=0), axis=0)
    lower_m = rep_m[np.newaxis, ...] - qh_m[:, np.newaxis, ...]
    upper_m = rep_m[np.newaxis, ...] - ql_m[:, np.newaxis, ...]

    # ----------------------------------------------------------------------------
    # Save outputs
    outdir = Path("UQ/conformal/results/ae_NNdzdt")
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{args.data_name}_cv+{args.folds}.pkl"
    with open(fname, "wb") as fp:
        pickle.dump(
            [
                args.alpha,
                args.test_size,
                outputs["idx_test"],
                (lower_D, upper_D, rep_D),
                (lower_m, upper_m, rep_m),
            ],
            fp,
        )
    print(f"Saved conformal results to {fname}.")


if __name__ == "__main__":
    main()
