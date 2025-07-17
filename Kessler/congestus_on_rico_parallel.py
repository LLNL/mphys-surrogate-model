"""
Parallel UQ with MAPIE-style conformal methods and SINDy.
Trains on congestus_test dataset, does everything else on rico_test dataset.
Two levels of concurrency:
  1) top_workers for jackknife / CV+ replicates.
  2) int_workers for integrating each IC in parallel.
"""

import argparse
import numpy as np
import pysindy as ps
import pickle
import os 
from threadpoolctl import threadpool_limits
from pysindy.optimizers import STLSQ
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from concurrent.futures import ProcessPoolExecutor, as_completed

# this automatically detects how the workers should be allocated based on the conformal predictions method used
def pick_workers(
    method_key,      # "full","split","jackknife","cv+"
    n_replicates,    # n_train for jk/jk+, k_folds for cv+
    total_cpus=None, # if None → from SLURM or os.cpu_count()
    override_top=None,
    override_int=None,
):
    # 1) detect total
    if total_cpus is None:
        total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count()

    # 2) user‐override?
    if override_top is not None and override_int is not None:
        return override_top, override_int

    # 3) methods without top‐level need no resampling
    if method_key in ("full", "split"):
        top, inte = 1, total_cpus

    else:
        # how many replicates/folds?
        R = n_replicates
        # want P × Q ≈ total_cpus
        P = min(R, max(1, int(np.sqrt(total_cpus))))
        Q = max(1, total_cpus // P)
        top, inte = P, Q

    return top, inte

# ── GLOBAL “templates” for library + optimizer ───────────────────────────
# Build these once in the main process, rely on fork() to share
threshold        = 1e-4
from pysindy.optimizers import STLSQ
optimizer_kwargs = dict(threshold=threshold)

# custom fractional‐exponent library (lambdas are inherited via fork)
def frac_xy(x, y):
    return x * np.abs(y)**0.875 * np.sign(y)

def frac_yx(x, y):
    return y * np.abs(x)**0.875 * np.sign(x)

def relu_minus(x):
    return np.maximum(0, x - 2.340384636957035)

def name_frac_xy(x, y):
    return f"{x} {y}^(0.875)"

def name_frac_yx(x, y):
    return f"{y} {x}^(0.875)"

def name_relu_minus(x):
    return f"max(0, {x}-a)"

library_functions      = [frac_xy, frac_yx, relu_minus]
library_function_names = [name_frac_xy, name_frac_yx, name_relu_minus]
custom_library         = ps.CustomLibrary(library_functions,
                                         library_function_names)
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False) + custom_library

def integrate_one(y0, t, model):
    """Integrate a single initial condition with fallback to damped RHS."""
    try:
        sol = solve_ivp(
            fun     = lambda t, y: model.predict(y.reshape(1, -1)).ravel(),
            t_span  = (t[0], t[-1]),
            y0      = y0,
            method  = "BDF",
            t_eval  = t,
            positive= True,
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
            fun     = lambda t, y: (
                model.predict(y.reshape(1, -1)).ravel() - eps * (y**5)
            ),
            t_span  = (t[0], t[-1]),
            y0      = y0,
            method  = "BDF",
            t_eval  = t,
            positive= True,
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
    lows  = _np.array(alpha_lows)
    ups   = 1.0 - _np.array(alpha_ups)
    all_q = _np.concatenate([lows, ups])       # e.g. [0.125, 0.025, 0.875, 0.975]

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
    q_low  = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high


# ─────────────────────────────────────────────────────────────────────────────
# Conformal‐UQ methods
# Each must return: (lower, upper, rep_traj)
#   - lower, upper: arrays of shape
#       (n_alpha, n_test, T, D)
#   - rep_traj: (n_test, T, D)
# ─────────────────────────────────────────────────────────────────────────────

# defined at module scope: leave-one-out worker for jackknife
def jk_loo_worker(i, model, states_train, times_train):
    """
    Top-level LOO worker: fit on all-but-i, then integrate the i-th trajectory 
    via integrate_one (no nested process pools).
    Returns (i, pred_i) where pred_i has shape (T, D).
    """
    n = states_train.shape[0]
    # 1) clone & fit on all-but-i
    m_i = clone(model)
    mask = np.arange(n) != i
    m_i.fit(
        list(states_train[mask]),
        t=list(times_train[mask]),
        multiple_trajectories=True
    )

    # 2) integrate left-out trajectory
    y0 = states_train[i, 0]
    t  = times_train[i]
    traj = integrate_one(y0=y0, t=t, model=m_i)

    # drop the initial t=0 state → shape (T, D)
    return i, traj[1:]

def compute_intervals_jackknife(model,
                                states_train, times_train,
                                states_test,  times_test,
                                alpha_lows, alpha_ups, 
                                int_workers):
    """
    JACKKNIFE (leave-one-out) in parallel across int_workers.
    """
    n = states_train.shape[0]
    # time–state dims (drop the initial timepoint)
    T = states_train.shape[1] - 1
    D = states_train.shape[2]
    
    # 1) build leave-one-out (LOO) predictions
    loo_preds = np.empty((n, T, D), dtype=float)
    # ensure ODE solver doesn't eat up other threads!
    with threadpool_limits(limits=1, user_api='openmp'): 
        with ProcessPoolExecutor(max_workers=min(int_workers, n)) as exe:
            futures = [
                exe.submit(jk_loo_worker,
                        i, model, states_train, times_train)
                for i in range(n)
            ]
            for fut in as_completed(futures):
                i, pred_i = fut.result()
                loo_preds[i] = pred_i
            
    # 2) signed residuals on train set 
    res_signed = loo_preds - states_train[:,1:]  

    # 3) one-sided quantiles per alpha 
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups) 

    # 4) full-data fit 
    model_full = clone(model)
    model_full.fit(list(states_train), t=list(times_train), multiple_trajectories=True)
    preds = evolve_states_parallel(
        model=model_full,
        initial_states=states_test[:,0], 
        times=times_test, 
        n_workers=int_workers
        )[:,1:]       

    # 5) asymmetric bounds on test set 
    # reshape so we have (n_alphas, 1, T, D) - (1, n_test, T, D)
    lower = preds[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = preds[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = preds # central prediction

    return lower, upper, rep_traj

def compute_intervals_full(model,
                           states_train, times_train,
                           states_test,  times_test,
                           alpha_lows, alpha_ups, int_workers):
    """
    FULL conformal: single fit, single calibrate on train,
    then test bounds.
    """
    # 1) fit & predict on training data 
    m = clone(model)
    m.fit(list(states_train), t=list(times_train), multiple_trajectories=True)
    pred_train = evolve_states_parallel(model=m, initial_states=states_train[:,0], times=times_train, n_workers=int_workers)[:,1:] 

    # 2) signed residuals on train set 
    res_signed = pred_train - states_train[:,1:]   

    # 3) one-sided quantiles per alpha
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups)    

    # 4) test predictions & bounds 
    pred_test = evolve_states_parallel(model=m, initial_states=states_test[:,0], times=times_test, n_workers=int_workers)[:,1:]   

    lower = pred_test[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = pred_test[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = pred_test
    
    return lower, upper, rep_traj

def compute_intervals_split(model,
                            states_train, times_train,
                            states_test,  times_test,
                            alpha_lows, alpha_ups,
                            p_percent, int_workers):
    """
    SPLIT conformal: split train→fit/calib, then test.
    """
    # 1) split training and calibration sets
    n = states_train.shape[0]
    split = int(n*(1 - 0.01*p_percent))
    idx_fit, idx_cal = train_test_split(
        np.arange(n),
        test_size=split,
        random_state=1952
    )
    times_fit  = times_train[idx_fit]
    times_cal   = times_train[idx_cal]
    states_fit = states_train[idx_fit]
    states_cal  = states_train[idx_cal]

    # 2) fit on states_fit, predict on states_cal
    m = clone(model)
    m.fit(list(states_fit), t=list(times_fit), multiple_trajectories=True)
    pred_cal = evolve_states_parallel(model=m, initial_states=states_cal[:,0], times=times_cal, n_workers=int_workers)[:,1:]   

    # 3) signed residuals on calibration set
    res_signed = pred_cal - states_cal[:,1:]      

    # 4) one-sided quantiles per alpha
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups) 

    # 5) test predictions & asymmetric bounds
    pred_test = evolve_states_parallel(model=m, initial_states=states_test[:,0],  times=times_test, n_workers=int_workers)[:,1:]   

    lower = pred_test[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = pred_test[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = pred_test
    
    return lower, upper, rep_traj

# 1) PER‐FOLD WORKER: no quantiles here, only residuals + test forecasts
def cvplus_worker(train_idx, val_idx,
                  model,
                  states_train, times_train,
                  states_test,  times_test,
                  int_workers):
    # 1a) fit on fold‐k train
    m_k = clone(model)
    m_k.fit(
        list(states_train[train_idx]),
        t=list(times_train[train_idx]),
        multiple_trajectories=True
    )

    # 1b) compute signed‐residuals on validation fold
    pred_val = evolve_states_parallel(
        model         = m_k,
        initial_states= states_train[val_idx, 0],
        times         = times_train[val_idx],
        n_workers     = 1
    )[:, 1:, :]                                          # (n_val, T-1, D)
    res_signed = pred_val - states_train[val_idx, 1:, :]    # (n_val, T-1, D)

    # 1c) predict on the test set
    pred_test = evolve_states_parallel(
        model         = m_k,
        initial_states= states_test[:, 0],
        times         = times_test,
        n_workers     = int_workers
    )[:, 1:, :]                                          # (n_test, T-1, D)

    return res_signed, pred_test


# 2) MAIN DRIVER: pool fold‐residuals → global quantiles, pool fold‐preds → median
def compute_intervals_cv_plus(model,
                              states_train, times_train,
                              states_test,  times_test,
                              alpha_lows, alpha_ups,
                              k_folds,
                              top_workers, int_workers):
    # 2a) launch CV‐fold workers in parallel
    kf    = KFold(n_splits=k_folds, shuffle=True, random_state=1952)
    folds = list(kf.split(states_train))

    all_res = []
    all_pred = []
    with ProcessPoolExecutor(max_workers=top_workers) as exe:
        futures = [
            exe.submit(
                cvplus_worker,
                tr_idx, va_idx,
                model,
                states_train, times_train,
                states_test, times_test,
                int_workers
            )
            for tr_idx, va_idx in folds
        ]
        for fut in as_completed(futures):
            res_signed, pred_test = fut.result()
            all_res.append(res_signed)
            all_pred.append(pred_test)

    # 2b) pool residuals across folds & compute global one‐sided quantiles
    resid_pool = np.concatenate(all_res, axis=0)              # (∑n_val, T, D)
    q_low, q_high = one_sided_quantiles(resid_pool, alpha_lows, alpha_ups)

    # 2c) pool per‐fold test forecasts & form median‐rep trajectory
    preds_stack = np.stack(all_pred, axis=0)                  # (k_folds, n_test, T, D)
    rep_traj    = np.median(preds_stack, axis=0)              # (n_test, T, D)

    # 2d) build global CV+ intervals
    #    lower = median_pred  -  high‐quantile(resid)
    #    upper = median_pred  -  low‐quantile(resid)
    lower = rep_traj[np.newaxis, ...] - q_high[:, np.newaxis, ...]
    upper = rep_traj[np.newaxis, ...] - q_low [:, np.newaxis, ...]

    return lower, upper, rep_traj

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", default="cv+5",
                        help="conformal method: jackknife, split[p], full, cv+[k].default is cv+5. \
                        The p in split indicates what *percent* you want to dedicate out of the training data for calibration, while\
                            the k in cv+ indicates how many models to train for cross-validation folds.")
    parser.add_argument("-a", "--alpha", nargs="+", type=float, default=[0.1],
                        help="significance level(s), in (0,1)")
    parser.add_argument("--top_workers", type=int,
                        help="override top-level parallelism (folds/replicates)")
    parser.add_argument("--int_workers", type=int,
                        help="override integration-level parallelism (gridboxes)")
    parser.add_argument("-c", "--cpus", type=int, default=None,
                        help="override total CPUs to use (default: None → SLURM or os.cpu_count())")
    args = parser.parse_args()

    # 2) Unpack & validate
    method = args.method
    k = None
    p = None
    if method.startswith("split"):
        p = float(method[5:])
        if not (0 < p < 100):
            raise ValueError("split[p]: p must be in (0,100)")
        method_key = "split"
    elif method.startswith("cv+"):
        k = int(method[3:])
        method_key = "cv+"
    else:
        method_key = method

    alphas = args.alpha
    if any(a <= 0 or a >= 1 for a in alphas):
        raise ValueError("alphas must be in (0,1)")
    alpha_lows = [a/2 for a in alphas]
    alpha_ups  = alpha_lows.copy()

    # 3a) Load congestus dataset as training set
    train_dset   = np.load('data/congestus_test_processed.npy')  
    times_train  = train_dset[..., 0]
    st_train = train_dset[..., 1:]
    
    # 3b) Load rico dataset as test set
    test_dset   = np.load('data/rico_test_processed.npy')  
    times_test  = test_dset[..., 0]
    st_test = test_dset[..., 1:]
    
    # 3c) Reduce number of timesteps so they match in each dataset
    n_t = min(st_train.shape[1], st_test.shape[1])
    times_train = times_train[:, :n_t]
    st_train = st_train[:, :n_t]
    times_test  = times_test[:, :n_t]
    st_test = st_test[:, :n_t]
    
    # 3d) Normalize both rico and congestus datasets
    # number of scalar‐elements in each set
    N1 = st_train.shape[0] * st_train.shape[1]
    N2 = st_test.shape[0] * st_test.shape[1]

    # per‐dataset mean over (0,1)
    mean_1 = st_train.mean(axis=(0,1))
    mean_2 = st_test.mean(axis=(0,1))

    # weighted combination to get the global mean
    means = (mean_1 * N1 + mean_2 * N2) / (N1 + N2)
    st_train /= means[None, None, :]
    st_test /= means[None, None, :]
    del train_dset, test_dset
    
    
    # 4) Decipher method and replicate count
    if method_key == "cv+":
        R = k # top-level = number of folds
    else:
        R = 1  # no top-level for full, split, or jackknife

    # -- pick workers --
    total_cpus = args.cpus # None if no argument passed
    top_workers, int_workers = pick_workers(
        method_key,
        n_replicates=R,
        total_cpus=total_cpus,
        override_top=args.top_workers,
        override_int=args.int_workers,
    )

    print(f"Using total_cpus={total_cpus!r}, top_workers={top_workers}, int_workers={int_workers}")

    # 5) Build SINDy model
    base_model = ps.SINDy(
        feature_library=feature_library,
        optimizer=STLSQ(**optimizer_kwargs)
    )

    # 6) Dispatch to the chosen method
    if method_key == "jackknife":
        lower, upper, rep = compute_intervals_jackknife(
            model=base_model,
            states_train=st_train, times_train=times_train,
            states_test=st_test,  times_test=times_test,
            alpha_lows=alpha_lows, alpha_ups=alpha_ups, 
            int_workers=int_workers  # how many concurrent LOO fits 
        )

    elif method_key == "full":
        lower, upper, rep = compute_intervals_full(
            model=base_model,
            states_train=st_train, times_train=times_train,
            states_test=st_test,  times_test=times_test,
            alpha_lows=alpha_lows, alpha_ups=alpha_ups,   
            int_workers=int_workers
        )

    elif method_key == "split":
        lower, upper, rep = compute_intervals_split(
            model=base_model,
            states_train=st_train, times_train=times_train,
            states_test=st_test,  times_test=times_test,
            alpha_lows=alpha_lows, alpha_ups=alpha_ups,
            p_percent=p, int_workers=int_workers
        )

    elif method_key == "cv+":
        lower, upper, rep = compute_intervals_cv_plus(
            model=base_model,
            states_train=st_train, times_train=times_train,
            states_test=st_test,  times_test=times_test,
            alpha_lows=alpha_lows, alpha_ups=alpha_ups,
            k_folds=k, top_workers=top_workers,   
            int_workers=int_workers
        )
    else:
        raise ValueError(f"Unknown method `{method}`")

    # 7) Save outputs
    outname = "data/congestus_on_rico" + f"/cp_{method_key}.pkl"
    with open(outname, "wb") as f:
        pickle.dump([alphas, lower, upper, rep], f)
    print(f"Saved conformal intervals → {outname}")


if __name__ == "__main__":
    main()