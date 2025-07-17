"""
main.py

Usage:
    echo "/path/to/your_processed.npy" | python main.py
"""

import os
os.environ["OMP_NUM_THREADS"]     = "1"
import sys
import numpy as np
import dill
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# 1) Worker‐side globals (populated by init_worker)
# -----------------------------------------------------------------------------
_times     = None
_states    = None
_model     = None
_final_eps = None
_out_dir   = None
_means     = None


def init_worker(file_path, model_file, final_eps, out_dir):
    """
    Called once in each worker. Memory-maps the dataset, normalizes it,
    and un-pickles the SINDy model.
    """
    global _times, _states, _means, _model, _final_eps, _out_dir

    # memory‐map dataset
    dset = np.load(file_path, mmap_mode="r")  # shape = (n_traj, n_steps, dim+1)
    _times  = dset[:, :, 0]
    _means = dset[:, :, 1:].mean(axis=(0,1)) 
    _states = dset[:, :, 1:] / _means[None, None, :]

    # load model (only once per worker)
    with open(model_file, "rb") as f:
        _model = dill.load(f)

    _final_eps = final_eps
    _out_dir   = out_dir


def process_trajectory(traj_no):
    """
    Integrate trajectory `traj_no`, plot the result, save to PDF,
    and return (traj_no, mode_used).
    """
    t    = _times[traj_no]
    true = _states[traj_no]
    y0   = true[0].copy()

    # pure‐SINDy RHS
    def fun(t_, x_):
        return _model.predict(x_.reshape(1, -1))[0]

    # damped RHS fallback
    def fun_damped(t_, x_):
        return fun(t_, x_) - _final_eps * (x_ ** 5)

    rtol, atol = 1e-6, 1e-8
    mode = "non-damped"

    try:
        sol = solve_ivp(
            fun      = fun,
            t_span   = (t[0], t[-1]),
            y0       = y0,
            method   = "BDF",
            t_eval   = t,
            positive = True,
            rtol     = rtol,
            atol     = atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        sol_y = sol.y.T

    except Exception:
        # fallback to damped
        mode = "damped"
        sol = solve_ivp(
            fun      = fun_damped,
            t_span   = (t[0], t[-1]),
            y0       = y0,
            method   = "BDF",
            t_eval   = t,
            positive = True,
            rtol     = rtol,
            atol     = atol,
        )
        if not sol.success:
            raise RuntimeError(f"Damped BDF failed: {sol.message}")
        sol_y = sol.y.T

    # denormalize all variables
    true *= _means[None, :]
    sol_y *= _means[None, :]

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, true[:, 0],  "b-o", markersize=3, label="Exact $q_c$")
    axs[0].plot(t, sol_y[:, 0],"r-x", markersize=3,
                label=f"SINDy ({mode}) $q_c$")
    axs[0].set_xlabel("t [seconds]")
    axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].grid(True)

    axs[1].plot(t, true[:, 1],  "b-o", markersize=3, label="Exact $q_r$")
    axs[1].plot(t, sol_y[:, 1],"r-x", markersize=3,
                label=f"SINDy ({mode}) $q_r$")
    axs[1].set_xlabel("t [seconds]")
    axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].grid(True)

    fig.suptitle(f"Trajectory {traj_no} ({mode})")
    fig.tight_layout()

    out_file = os.path.join(_out_dir, f"trajectory{traj_no}.pdf")
    fig.savefig(out_file)
    plt.close(fig)

    return traj_no, mode


# -----------------------------------------------------------------------------
# 2) Main: only executed in the main process
# -----------------------------------------------------------------------------
def main():
    # Switch to 'fork' on macOS/Linux to avoid re-importing the module
    mp.set_start_method("fork", force=True)

    # Read the numpy‐file path once, in main only
    file_path = sys.stdin.readline().strip()
    if not file_path:
        print("ERROR: expected a .npy path on stdin", file=sys.stderr)
        sys.exit(1)

    # Build derived paths
    model_file = file_path.removesuffix("processed.npy") + "sindy_model.pkl"
    out_dir    = file_path.removesuffix("_processed.npy") + "/SINDy"
    os.makedirs(out_dir, exist_ok=True)

    # Memory‐map just to compute the normalization
    dset  = np.load(file_path, mmap_mode="r")

    # Compute final_eps from model coefficients
    with open(model_file, "rb") as f:
        tmp = dill.load(f)
    abs_coefs = np.abs(tmp.coefficients()).ravel()
    base_eps  = 0.01 * 1e-5
    final_eps = base_eps * np.min(abs_coefs[abs_coefs > 0])
    del tmp

    # Spawn a Pool: workers will memmap the .npy and load the model once
    n_traj  = dset.shape[0]
    init_args = (file_path, model_file, final_eps, out_dir)
    with mp.Pool(
        processes=mp.cpu_count(),
        initializer = init_worker,
        initargs     = init_args
    ) as pool:
        results = pool.map(process_trajectory, range(n_traj))

    # Report
    print("Finished all trajectories:")
    for traj_no, mode in results:
        print(f" • Traj {traj_no}: used {mode}")


if __name__ == "__main__":
    main()