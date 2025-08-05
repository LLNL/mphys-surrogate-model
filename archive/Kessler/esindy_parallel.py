"""
esindy_parallel.py

Perform bootstrap-sampling fits of a SINDy model in parallel,
using ProcessPoolExecutor.  User can specify:

  --input        Path to processed .npy file
  --threshold    STLSQ sparsity threshold (float)
  --frac         Fraction of total samples to bootstrap (default=0.05)
  --max-workers  Max parallel workers (default=os.cpu_count())
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import pysindy as ps
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from concurrent.futures import ProcessPoolExecutor
import pickle

# -----------------------------------------------------------------------------
# WORKER‐LEVEL GLOBALS (populated via init_worker)
# -----------------------------------------------------------------------------
_states = None
_derivatives = None
_feature_library = None
_optimizer = None


def build_feature_library(threshold):
    """
    Build the same feature library & STLSQ optimizer
    for a given sparsity threshold.
    """
    lib_funcs = [
        lambda x, y: x * np.abs(y) ** 0.875 * np.sign(y),
        lambda x, y: y * np.abs(x) ** 0.875 * np.sign(x),
        lambda x: np.maximum(0, x - 2.340384636957035),
    ]
    lib_names = [
        lambda x, y: f"{x} {y}^(0.875)",
        lambda x, y: f"{y} {x}^(0.875)",
        lambda x: f"max(0,{x}-a)",
    ]
    custom_lib = ps.CustomLibrary(
        library_functions=lib_funcs,
        function_names=lib_names,
    )
    poly_lib = ps.PolynomialLibrary(degree=3, include_bias=False)
    feature_lib = poly_lib + custom_lib
    optimizer = STLSQ(threshold=threshold)
    return feature_lib, optimizer


def init_worker(states, derivatives, threshold):
    """
    Initializer for each worker process:
      - stash the big arrays
      - rebuild feature library & optimizer
    """
    global _states, _derivatives, _feature_library, _optimizer
    _states = states
    _derivatives = derivatives
    _feature_library, _optimizer = build_feature_library(threshold)


def bootstrap_worker(seed):
    """
    Each worker:
      - draws a bootstrap sample
      - fits a fresh SINDy model
      - returns its coefficient matrix
    """
    import numpy as _np
    import pysindy as _ps

    rng = _np.random.RandomState(seed)
    idx = rng.randint(0, len(_states), size=len(_states))

    model = _ps.SINDy(feature_library=_feature_library, optimizer=_optimizer)
    model.fit(_states[idx], x_dot=_derivatives[idx])
    return model.coefficients()


def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel bootstrap sampling for SINDy coefficient estimation"
    )
    p.add_argument("file_path", help=".npy dataset")
    p.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=1e-4,
        help="Sparsity threshold for STLSQ (default=1e-4)",
    )
    p.add_argument(
        "--frac",
        "-f",
        type=float,
        default=0.05,
        help="Fraction of total samples to bootstrap (default=0.05)",
    )
    p.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=None,
        help="Max parallel workers (default=os.cpu_count())",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load dataset
    dset = np.load(args.file_path)  # shape: (n_traj, n_tsteps, n_vars+1)

    # 2) Split times & states, normalize
    times_list = list(dset[:, :, 0])  # list of length n_traj, each (n_tsteps,)
    states_arr = dset[:, :, 1:]  # shape (n_traj, n_tsteps, n_vars)
    means = states_arr.mean(axis=(0, 1))
    states_arr /= means[None, None, :]
    states_list = list(states_arr)

    # 3) Compute derivatives
    derivatives = np.empty(states_arr.shape, dtype=float)  # same shape as states_arr
    for i, t in enumerate(times_list):
        derivatives[i] = ps.differentiation.FiniteDifference()._differentiate(
            states_list[i], times_list[i]
        )

    # 4) Flatten for pre‐fit
    states_flat = np.vstack(states_list)
    derivatives_flat = np.vstack(derivatives)

    # 5) Pre‐fit to detect features
    base_lib, base_opt = build_feature_library(args.threshold)
    model0 = ps.SINDy(feature_library=base_lib, optimizer=base_opt)
    model0.fit(states_flat, x_dot=derivatives_flat)
    features = model0.get_feature_names()
    print(
        f"Detected {model0.n_output_features_} features "
        f"for {states_flat.shape[1]} state variables."
    )

    # 6) Determine number of bootstraps
    q = int(np.round(args.frac * len(states_flat)))
    if q < 1:
        raise ValueError("Fraction too small: results in zero bootstrap samples")
    seeds = list(range(q))
    print(f"Launching {q} bootstrap fits using threshold={args.threshold}")

    # 7) Parallel bootstrap fits
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(states_flat, derivatives_flat, args.threshold),
    ) as exe:
        coeff_samples = list(exe.map(bootstrap_worker, seeds))

    bootstraps = np.stack(coeff_samples, axis=0)
    print("Bootstraps complete; shape =", bootstraps.shape)

    # 8) Save results
    base = args.file_path.removesuffix("_processed.npy")
    out_file = f"{base}_esindy.pkl"
    with open(out_file, "wb") as f:
        pickle.dump((bootstraps, features), f)
    print("Saved coefficients + feature names to:", out_file)


if __name__ == "__main__":
    main()
