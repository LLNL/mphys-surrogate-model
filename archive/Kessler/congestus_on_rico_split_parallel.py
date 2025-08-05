"""
Parallel UQ with MAPIE-style conformal methods and SINDy.
Trains on congestus_test dataset, does everything else on rico_test dataset.
Concurrency only over int_workers for integrating each IC in parallel.
"""

import argparse
import numpy as np
import pysindy as ps
import pickle
import os
from pysindy.optimizers import STLSQ
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── GLOBAL “templates” for library + optimizer ───────────────────────────
# Build these once in the main process, rely on fork() to share
threshold = 1e-4
from pysindy.optimizers import STLSQ

optimizer_kwargs = dict(threshold=threshold)


# custom fractional‐exponent library (lambdas are inherited via fork)
def frac_xy(x, y):
    return x * np.abs(y) ** 0.875 * np.sign(y)


def frac_yx(x, y):
    return y * np.abs(x) ** 0.875 * np.sign(x)


def relu_minus(x):
    return np.maximum(0, x - 2.340384636957035)


def name_frac_xy(x, y):
    return f"{x} {y}^(0.875)"


def name_frac_yx(x, y):
    return f"{y} {x}^(0.875)"


def name_relu_minus(x):
    return f"max(0, {x}-a)"


library_functions = [frac_xy, frac_yx, relu_minus]
library_function_names = [name_frac_xy, name_frac_yx, name_relu_minus]
custom_library = ps.CustomLibrary(library_functions, library_function_names)
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False) + custom_library


def integrate_one(y0, t, model):
    """Integrate a single initial condition with fallback to damped RHS."""
    try:
        sol = solve_ivp(
            fun=lambda t, y: model.predict(y.reshape(1, -1)).ravel(),
            t_span=(t[0], t[-1]),
            y0=y0,
            method="BDF",
            t_eval=t,
            positive=True,
        )
        sol_success = sol.success
    except Exception:
        sol_success = False

    if not sol_success:
        # fallback to damped RHS
        abs_coefs = np.abs(model.coefficients())
        min_nonzero = np.min(abs_coefs[abs_coefs > 0]).flatten()
        eps = 1e-7 * min_nonzero
        sol = solve_ivp(
            fun=lambda t, y: (model.predict(y.reshape(1, -1)).ravel() - eps * (y**5)),
            t_span=(t[0], t[-1]),
            y0=y0,
            method="BDF",
            t_eval=t,
            positive=True,
        )
        if not sol.success:
            raise RuntimeError(f"Damped integration failed: {sol.message}")

    return sol.y.T  # shape (n_t, n_vars)


def evolve_states_parallel(model, initial_states, times, n_workers):
    """
    Parallel integration across many initial conditions.
      - initial_states: (n_ic, n_vars)
      - times: either shape (n_ic, n_t) or single (n_t,)
    Returns predicted_states: (n_ic, n_t, n_vars)
    """
    # Broadcast times if needed
    if times.ndim == 1:
        time_list = [times] * len(initial_states)
    else:
        time_list = [times[i] for i in range(len(initial_states))]

    preds = [None] * len(initial_states)
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {
            exe.submit(integrate_one, initial_states[i], time_list[i], model): i
            for i in range(len(initial_states))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            preds[idx] = fut.result()

    return np.stack(preds, axis=0)


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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # 1) Parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--cal_size",
        required=True,
        help="the fraction of RICO data you want to dedicate to calibration.",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        nargs="+",
        type=float,
        default=[0.1],
        help="significance level(s), in (0,1)",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=None,
        help="override total CPUs to use (default: None → SLURM or os.cpu_count())",
    )
    args = parser.parse_args()

    # 2) Unpack & validate
    cal_size = float(args.cal_size)
    if not (0 < cal_size < 1):
        raise ValueError("calibration size must be in (0,1)")
    alphas = args.alpha
    if any(a <= 0 or a >= 1 for a in alphas):
        raise ValueError("alphas must be in (0,1)")
    alpha_lows = [a / 2 for a in alphas]
    alpha_ups = alpha_lows.copy()

    # 3a) Load congestus dataset as training set
    congestus_dset = np.load("data/congestus_test_processed.npy")
    times_congestus = congestus_dset[..., 0]
    st_congestus = congestus_dset[..., 1:]

    # 3b) Load rico dataset as calibration + test set
    rico_dset = np.load("data/rico_test_processed.npy")
    times_rico = rico_dset[..., 0]
    st_rico = rico_dset[..., 1:]

    # 3c) Reduce number of timesteps so they match in each dataset
    n_t = min(st_congestus.shape[1], st_rico.shape[1])
    times_congestus = times_congestus[:, :n_t]
    st_congestus = st_congestus[:, :n_t]
    times_rico = times_rico[:, :n_t]
    st_rico = st_rico[:, :n_t]

    # 3d) Normalize both rico and congestus datasets
    # number of scalar‐elements in each set
    N1 = st_congestus.shape[0] * st_congestus.shape[1]
    N2 = st_rico.shape[0] * st_rico.shape[1]

    # per‐dataset mean over (0,1)
    mean_1 = st_congestus.mean(axis=(0, 1))
    mean_2 = st_rico.mean(axis=(0, 1))

    # weighted combination to get the global mean
    means = (mean_1 * N1 + mean_2 * N2) / (N1 + N2)
    st_congestus /= means[None, None, :]
    st_rico /= means[None, None, :]
    del congestus_dset, rico_dset

    # 4) split RICO into calibration and test sets

    idx_cal, idx_test = train_test_split(
        np.arange(len(st_rico)), test_size=1 - cal_size, random_state=1952
    )
    times_cal = times_rico[idx_cal]
    times_test = times_rico[idx_test]
    states_cal = st_rico[idx_cal]
    states_test = st_rico[idx_test]

    # 5) Pick workers
    total_cpus = args.cpus  # None if no argument passed
    if total_cpus is None:
        total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()

    print(f"Using total_cpus={total_cpus!r}")

    # 6) Build SINDy model
    base_model = ps.SINDy(
        feature_library=feature_library, optimizer=STLSQ(**optimizer_kwargs)
    )

    # 7) Run conformal method
    # This is split-conformal with congestus as the training set and RICO split across the calibration and testing sets.
    # 7a) fit on states_fit, predict on states_cal
    m = clone(base_model)
    m.fit(list(st_congestus), t=list(times_congestus), multiple_trajectories=True)
    pred_cal = evolve_states_parallel(
        model=m, initial_states=states_cal[:, 0], times=times_cal, n_workers=total_cpus
    )[:, 1:]

    # 7b) signed residuals on calibration set
    res_signed = pred_cal - states_cal[:, 1:]

    # 7c) one-sided quantiles per alpha
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups)

    # 7d) test predictions & asymmetric bounds
    pred_test = evolve_states_parallel(
        model=m,
        initial_states=states_test[:, 0],
        times=times_test,
        n_workers=total_cpus,
    )[:, 1:]

    lower = pred_test[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = pred_test[np.newaxis, :, :, :] - q_low[:, np.newaxis, :, :]

    # 8) Save outputs
    outname = "data/congestus_on_rico_split" + f"/cp.pkl"
    with open(outname, "wb") as f:
        pickle.dump([alphas, idx_test, lower, upper, pred_test], f)
    print(f"Saved conformal intervals → {outname}")


if __name__ == "__main__":
    main()
