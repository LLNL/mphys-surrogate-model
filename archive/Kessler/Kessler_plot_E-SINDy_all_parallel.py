import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"  # ensure ODE solver doesn't eat up other threads
import numpy as np
import pysindy as ps
from fastdigest import TDigest
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor

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


def fun_damped(model, t, x):
    """
    Fifth-order damping term scaled by ε relative to the smallest nonzero coefficient.
    """
    eps = 0.01 * 10 ** (-5)
    abs_coefs = np.abs(model.coefficients())
    final_eps = eps * np.min(abs_coefs[abs_coefs > 0]).flatten()

    x_eval = x.reshape(1, -1)
    dxdt = model.predict(x_eval) - final_eps * (x_eval**5)
    return dxdt[0]


# ── “initializer” for each worker ───────────────────────────────────────
def init_worker(flat_states, derivatives, times_list, init_states, quantile_query):
    global _X, _dX, _times_list, _init_states, _Q
    global _n_vars, _quantile_query

    _X = flat_states
    _dX = derivatives
    _times_list = times_list
    _init_states = init_states
    _quantile_query = quantile_query

    # number of state variables
    _n_vars = flat_states.shape[1]


# ── what each worker does ────────────────────────────────────────────────
def run_bootstrap(rep_id):
    # 1) bootstrap‐resample
    idx = np.random.randint(0, _X.shape[0], size=_X.shape[0])
    Xs, dXs = _X[idx], _dX[idx]

    # 2) rebuild & fit a fresh SINDy model
    model = ps.SINDy(
        feature_library=feature_library, optimizer=STLSQ(**optimizer_kwargs)
    )
    model.fit(Xs, x_dot=dXs)

    # 3) prepare a local array of T-Digests: one digest per (traj, time, var)
    M = len(_times_list)
    local_tds = []
    for j in range(M):
        Tj = len(_times_list[j])
        arr = np.empty((Tj, _n_vars), dtype=object)
        for t in range(Tj):
            for v in range(_n_vars):
                arr[t, v] = TDigest()
        local_tds.append(arr)

    # 4) simulate each trajectory & update digests
    for j in range(M):
        t_j = _times_list[j]
        y0 = _init_states[j].copy()

        # 1) Try original RHS

        try:
            sol = solve_ivp(
                fun=lambda t, x: model.predict(x.reshape(1, -1))[0],
                t_span=(t_j[0], t_j[-1]),
                y0=y0,
                method="BDF",
                t_eval=t_j,
                positive=True,
            )
            if not sol.success:
                raise RuntimeError(f"Original solve failed: {sol.message}")

        except Exception as e_orig:
            # 2) Fallback to damped RHS
            sol = solve_ivp(
                fun=lambda t, x: fun_damped(model, t, x),
                t_span=(t_j[0], t_j[-1]),
                y0=y0,
                method="BDF",
                t_eval=t_j,
                positive=True,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Trajectory {j}: both original and damped solves failed.\n"
                    f"  - Original error: {e_orig}\n"
                    f"  - Damped message: {sol.message}"
                )

        # 3) Transpose & clamp
        sol_y = sol.y.T  # shape (len(t_j), _n_vars)
        np.maximum(sol_y, 0, out=sol_y)  # enforce non-negativity if needed

        # 4) Bulk‐update TDigest objects
        for (t_idx, v_idx), value in np.ndenumerate(sol_y):
            local_tds[j][t_idx, v_idx].update(value)

    return local_tds


# ── merge B workers into one final structure ─────────────────────────────
def merge_tdigests(all_reps_tds):
    M = len(all_reps_tds[0])
    final = []
    for j in range(M):
        Tj, nv = all_reps_tds[0][j].shape
        merged = np.empty((Tj, nv), dtype=object)
        for t in range(Tj):
            for v in range(nv):
                # start from the first rep
                merged[t, v] = all_reps_tds[0][j][t, v]
                # then “re-assign” the merged result
                for rep in all_reps_tds[1:]:
                    merged[t, v] = merged[t, v].merge(rep[j][t, v])
        final.append(merged)
    return final


# ── compute your quantiles ──────────────────────────────────────────────
def compute_quantiles(final_tds, quantile_query):
    M = len(final_tds)
    Q = len(quantile_query)
    out = []
    for j in range(M):
        Tj, nv = final_tds[j].shape
        q_arr = np.full((Tj, nv, Q), np.nan)
        for t in range(Tj):
            for v in range(nv):
                td = final_tds[j][t, v]
                q_arr[t, v, :] = [td.quantile(q) for q in quantile_query]
        out.append(q_arr)
    return out


# ── detect number of jobs ────────────────────────────────────────────────
def detect_n_jobs(cli_n_jobs=None):
    if cli_n_jobs:
        return cli_n_jobs
    # 1) Slurm “cpus-per-task”
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    # 2) optional: SLURM_JOB_CPUS_PER_NODE
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        # sometimes formatted as “55(x1)”, so strip non‐digits
        return int("".join(filter(str.isdigit, os.environ["SLURM_JOB_CPUS_PER_NODE"])))
    # 3) fallback to all local cores
    return multiprocessing.cpu_count()


# ── main driver ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("file_path", help=".npy dataset")
    p.add_argument("-B", "--boots", type=int, required=True)
    p.add_argument("-t", "--traj_nos", nargs="+", type=int, required=True)
    p.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        help="override # of workers (else detect from Slurm)",
    )
    args = p.parse_args()

    n_jobs = detect_n_jobs(args.n_jobs)
    print(f"→ launching with n_jobs = {n_jobs}")

    # load & normalize
    dset = np.load(args.file_path)
    times = dset[..., 0]
    states = dset[..., 1:]
    var_means = states.mean(axis=(0, 1))
    states = (
        states / var_means[None, None, :]
    )  # normalize each variable by its own mean

    # compute derivatives & flatten
    ders = [ps.FiniteDifference()._differentiate(s, t) for s, t in zip(states, times)]
    flat_states = np.vstack(states)
    flat_ders = np.vstack(ders)

    # select only your traj_nos
    traj_nos = np.array(args.traj_nos)
    times_list = [times[k] for k in traj_nos]
    init_list = [states[k, 0, :] for k in traj_nos]

    # quantiles to query
    quantile_query = 0.01 * np.array([1, 2.5, 5, 50, 95, 97.5, 99])

    # ─── launch B tasks in parallel ───────────────────────────────────────
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=init_worker,
        initargs=(flat_states, flat_ders, times_list, init_list, quantile_query),
    ) as exe:
        all_reps = list(exe.map(run_bootstrap, range(args.boots)))

    # merge & compute CI
    final_tds = merge_tdigests(all_reps)
    # # check digests non-empty
    # for j, arr in enumerate(final_tds):
    #     nonempty = sum(1 for td in arr.ravel() if td.n_values>0)
    #     print(f"[After merge] traj {j}: {nonempty}/{arr.size} digests nonempty")
    quantiles = compute_quantiles(final_tds, quantile_query)

    # we don't need the full derivatives or states anymore
    del flat_states, flat_ders

    # denormalize everything for plotting
    states *= var_means[None, None, :]
    quantiles *= var_means[None, None, :, None]

    # WARNING: the rest of the script is a lot less mutable than the parts above

    import matplotlib

    matplotlib.use("Agg")  # no GUI backend
    import matplotlib.pyplot as plt

    # loop through trajectories to compute quantiles, then make plots
    M = len(final_tds)
    for j in range(M):
        t = times_list[j]
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # First subplot: q_c
        axs[0].plot(
            t,
            states[traj_nos[j]][:, 0],
            color="blue",
            marker="o",
            markersize=0.5,
            label=r"Exact, $q_c$",
        )
        axs[0].fill_between(
            t,
            quantiles[j][:, 0, 0],
            quantiles[j][:, 0, 6],
            color="yellow",
            alpha=0.3,
            label="98% CI",
        )
        axs[0].fill_between(
            t,
            quantiles[j][:, 0, 1],
            quantiles[j][:, 0, 5],
            color="orange",
            alpha=0.3,
            label="95% CI",
        )
        axs[0].fill_between(
            t,
            quantiles[j][:, 0, 2],
            quantiles[j][:, 0, 4],
            color="red",
            alpha=0.3,
            label="90% CI",
        )
        axs[0].plot(
            t,
            quantiles[j][:, 0, 3],
            color="red",
            marker="x",
            markersize=0.5,
            label=r"SINDy (median), $q_c$",
        )
        axs[0].set_xlabel("t [seconds]")
        axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[0].grid(True)

        # Second subplot: q_r
        axs[1].plot(
            t,
            states[traj_nos[j]][:, 1],
            color="blue",
            marker="o",
            markersize=0.5,
            label=r"Exact, $q_r$",
        )
        axs[1].fill_between(
            t,
            quantiles[j][:, 1, 0],
            quantiles[j][:, 1, 6],
            color="yellow",
            alpha=0.3,
            label="98% CI",
        )
        axs[1].fill_between(
            t,
            quantiles[j][:, 1, 1],
            quantiles[j][:, 1, 5],
            color="orange",
            alpha=0.3,
            label="95% CI",
        )
        axs[1].fill_between(
            t,
            quantiles[j][:, 1, 2],
            quantiles[j][:, 1, 4],
            color="red",
            alpha=0.3,
            label="90% CI",
        )
        axs[1].plot(
            t,
            quantiles[j][:, 1, 3],
            color="red",
            marker="x",
            markersize=0.5,
            label=r"SINDy (median), $q_r$",
        )
        axs[1].set_xlabel("t [seconds]")
        axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[1].grid(True)

        fig.suptitle(f"Trajectory No. {traj_nos[j]}")
        plt.tight_layout()
        plt.savefig(
            args.file_path.removesuffix("_processed.npy")
            + "/E-SINDy/CI/intervals"
            + str(traj_nos[j])
            + ".pdf"
        )
        plt.close()


if __name__ == "__main__":
    main()
