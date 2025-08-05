# -------------------------------------------------------------------
#  1) Force spawn BEFORE torch/CUDA imports
#  2) Limit BLAS threads to 1 to avoid oversubscription
# -------------------------------------------------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import multiprocessing as _mp

_mp.set_start_method("spawn", force=True)

# -------------------------------------------------------------------
#  Standard imports
# -------------------------------------------------------------------
import sys
import argparse
import time
import pickle
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import torch

# -------------------------------------------------------------------
#  If you need to import from src/ and training_scripts/, adjust sys.path
# -------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import training
from training_scripts import train_ae_ar as train

# -------------------------------------------------------------------
#  Argument parsing
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run cv+ conformal predictions for AE-AR")
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t",
    "--test_size",
    type=float,
    default=0.2,
    help="proportion for test split (0 < t < 1), default 0.2",
)
parser.add_argument(
    "-k",
    "--folds",
    type=int,
    default=5,
    help="number of cross-validation folds. default is 5",
)
parser.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=None,
    help="override number of CPUs (default: SLURM_CPUS_PER_TASK or os.cpu_count())",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=100, help="number of epochs (default 100)"
)
parser.add_argument(
    "-b", "--batches", type=int, default=200, help="batch size (default 200)"
)
parser.add_argument(
    "-a",
    "--alpha",
    nargs="+",
    type=float,
    default=[0.1],
    help="miscoverage rate(s), each in (0,1), default [0.1]",
)
args = parser.parse_args()

# model & training hyper‐parameters
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
    "n_bins": None,  # set after data load
}

torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])

# load the dataset
sample_time = None
outputs = du.open_mass_dataset(
    name=args.data_name,
    data_dir=Path(__file__).parent.parent.parent / "data",
    sample_time=sample_time,
    test_size=args.test_size,
    calib_size=None,
    random_state=params["random_seed"],
)
params["n_bins"] = outputs["n_bins"]

# alphas & tails
alphas = args.alpha
if any(a <= 0 or a >= 1 for a in alphas):
    raise ValueError("Alpha must lie in (0,1).")
alpha_lows = [a / 2 for a in alphas]
alpha_ups = alpha_lows.copy()


# -------------------------------------------------------------------
#  Initialization utilities
# -------------------------------------------------------------------
def init_model(device, params):
    model = train.AEAutoregressor(
        n_channels=1,
        n_bins=params["n_bins"],
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
        "erf_FFNN_latent3_order(63, 98, 30)_tr1000_"
        "lr0.002482884780966882_bs4_"
        "weights0.22816989332325596-0.6719555656053005_"
        "7c43ff3e659b47358fa327f690871ed7",
    )
    ckpt = torch.load(
        os.path.join(
            optimal_path,
            "erf_FFNN_latent3_order(63, 98, 30)_tr1000_"
            "lr0.002482884780966882_bs4_"
            "weights0.22816989332325596-0.6719555656053005_"
            "7c43ff3e659b47358fa327f690871ed7.pth",
        ),
        weights_only=True,
    )
    model.load_state_dict(ckpt)
    return model.to(device)


def init_optimizer(model, params):
    return torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["wd"]
    )


def init_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")


# -------------------------------------------------------------------
#  Conformal‐helper: compute one‐sided quantiles for lower/upper
# -------------------------------------------------------------------
def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    lows = np.array(alpha_lows)
    ups = 1.0 - np.array(alpha_ups)
    all_q = np.concatenate([lows, ups])
    sort_i = np.argsort(all_q)
    qs = np.quantile(residuals, all_q[sort_i], axis=0)
    unsort = np.empty_like(qs)
    unsort[sort_i] = qs
    m = len(alpha_lows)
    return unsort[:m], unsort[m:]


# -------------------------------------------------------------------
#  Run three‐part AE‐AR architecture on each batch
#    returns [ decoder_only, latent_only, full ], mass‐trajectories
# -------------------------------------------------------------------
def run_all(x, m, model, params):
    n_lag = params["n_lag"]
    # 1) decoder-only
    D0 = model.decoder(model.encoder(torch.Tensor(x))).detach().numpy()

    # 2) latent-only
    L1 = np.empty((x.shape[0], x.shape[1], params["latent_dim"]), dtype=float)
    for i in range(x.shape[0]):
        # initialize first n_lag
        enc0 = model.encoder(torch.Tensor(x[i, :n_lag, :])).detach().numpy()
        buf = np.empty((x.shape[1], params["latent_dim"] + 1))
        buf[:n_lag, :-1] = enc0
        buf[:n_lag, -1] = m[i, :n_lag]
        # roll forward in latent
        for t in range(n_lag, x.shape[1]):
            inp = torch.Tensor(buf[t - n_lag : t].reshape(1, n_lag, buf.shape[1]))
            res = model.autoregressor(inp).detach().numpy().squeeze()
            buf[t] = res
        L1[i] = buf[:, :-1]

    # 3) full auto‐regressive + decoder
    D2 = np.empty_like(x)
    M2 = np.empty_like(m)
    # seed first n_lag
    D2[:, :n_lag, :] = D0[:, :n_lag, :]
    M2[:, :n_lag] = m[:, :n_lag]
    for i in range(x.shape[0]):
        for t in range(n_lag, x.shape[1]):
            enc = model.encoder(
                torch.Tensor(D2[i, t - n_lag : t, :].reshape(1, n_lag, x.shape[2]))
            )
            mass_in = torch.Tensor(M2[i, t - n_lag : t].reshape(1, n_lag, 1))
            res = model.autoregressor(torch.cat([enc, mass_in], dim=2))
            D2[i, t, :] = model.decoder(res[..., :-1]).detach().numpy()
            M2[i, t] = res[..., -1].detach().item()

    return [D0, L1, D2], M2


# -------------------------------------------------------------------
#  Worker for a single fold – re‐inits model/training entirely
# -------------------------------------------------------------------
def _fold_worker(args):
    fold_id, train_idx, val_idx, args_ns, params, outputs, device = args

    # re‐init everything inside child
    model = init_model(device, params)
    optimizer = init_optimizer(model, params)
    sched = init_scheduler(optimizer)

    # build loaders
    train_ds = du.NormedBinDatasetAR(
        outputs["x_train"][train_idx],
        outputs["m_train"][train_idx],
        lag=params["n_lag"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True
    )
    test_ds = du.NormedBinDatasetAR(
        outputs["x_test"], outputs["m_test"], lag=params["n_lag"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False
    )

    # silent‐train
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
        best_model, *_ = train.train_and_eval(
            params["num_epochs"],
            model,
            train_loader,
            test_loader,
            optimizer,
            sched,
            params,
            early_stopping=training.EarlyStopping(patience=params["patience"]),
            print_flag=False,
            device=device,
        )

    # run on calib (the val‐split) + test
    D_calib, M_calib = run_all(
        outputs["x_train"][val_idx], outputs["m_train"][val_idx], best_model, params
    )
    D_test, M_test = run_all(outputs["x_test"], outputs["m_test"], best_model, params)

    # collect residuals & predictions
    D_res = []
    D_pred = []
    for i in range(3):
        if i == 1:
            enc_val = (
                best_model.encoder(torch.Tensor(outputs["x_train"][val_idx]).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            resid = D_calib[i] - enc_val
        else:
            resid = D_calib[i] - outputs["x_train"][val_idx]
        D_res.append(resid)
        D_pred.append(D_test[i])

    M_resid = M_calib - outputs["m_train"][val_idx]
    M_pred = M_test

    print(f"[Fold {fold_id+1}] done.")
    return D_res, D_pred, M_resid, M_pred


# -------------------------------------------------------------------
#  main() only implements cv+ branch
# -------------------------------------------------------------------
def main():
    # how many workers?
    total_cpus = (
        args.cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()
    )

    # pick device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available() and params["batch_size"] > 1000
            else "cpu"
        )
    )
    print(f"Using device: {device}, workers={total_cpus}, folds={args.folds}")

    # build CV splits
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=params["random_seed"])
    splits = list(kf.split(outputs["x_train"]))

    # prepare fold arguments
    fold_args = [
        (fold_id, train_idx, val_idx, args, params, outputs, device)
        for fold_id, (train_idx, val_idx) in enumerate(splits)
    ]

    # ----------------------------------------------------------------
    #  Launch spawn‐based pool
    # ----------------------------------------------------------------
    from multiprocessing import get_context

    ctx = get_context("spawn")
    with ctx.Pool(processes=total_cpus) as pool:
        all_results = pool.map(_fold_worker, fold_args)

    # ----------------------------------------------------------------
    #  Aggregate residuals & predictions
    # ----------------------------------------------------------------
    DSD_resid = [[], [], []]
    DSD_pred = [[], [], []]
    M_resid = []
    M_pred = []
    for D_res, D_p, m_res, m_p in all_results:
        for i in range(3):
            DSD_resid[i].append(D_res[i])
            DSD_pred[i].append(D_p[i])
        M_resid.append(m_res)
        M_pred.append(m_p)

    # ----------------------------------------------------------------
    #  Compute global conformal intervals
    # ----------------------------------------------------------------
    rep_DSD = [None] * 3
    lower = [
        np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),
        np.empty(
            (
                len(alphas),
                outputs["x_test"].shape[0],
                outputs["x_test"].shape[1],
                params["latent_dim"],
            ),
            dtype=float,
        ),
        np.empty((len(alphas),) + outputs["x_test"].shape, dtype=float),
    ]
    upper = [arr.copy() for arr in lower]

    for i in range(3):
        pool_res = np.concatenate(DSD_resid[i], axis=0)
        ql, qh = one_sided_quantiles(pool_res, alpha_lows, alpha_ups)
        center = np.median(np.stack(DSD_pred[i], axis=0), axis=0)
        rep_DSD[i] = center
        lower[i] = center[np.newaxis, ...] - qh[:, np.newaxis, ...]
        upper[i] = center[np.newaxis, ...] - ql[:, np.newaxis, ...]

    # mass‐trajectories
    pool_m = np.concatenate(M_resid, axis=0)
    ql_m, qh_m = one_sided_quantiles(pool_m, alpha_lows, alpha_ups)
    rep_m = np.median(np.stack(M_pred, axis=0), axis=0)
    lower_m = rep_m[np.newaxis, ...] - qh_m[:, np.newaxis, ...]
    upper_m = rep_m[np.newaxis, ...] - ql_m[:, np.newaxis, ...]

    # ----------------------------------------------------------------
    #  Save everything
    # ----------------------------------------------------------------
    out_dir = Path("UQ/conformal/results/ae_ar")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{args.data_name}_cv+{args.folds}.pkl"
    with open(fname, "wb") as fh:
        pickle.dump(
            [
                alphas,
                outputs["idx_test"],
                (lower, upper, rep_DSD),
                (lower_m, upper_m, rep_m),
            ],
            fh,
        )
    print(f"Saved conformal results to {fname}.")


if __name__ == "__main__":
    main()
