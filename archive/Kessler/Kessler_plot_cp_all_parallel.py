"""
Parallel plotting of conformal-interval trajectories
using concurrent.futures.ProcessPoolExecutor.
"""

import os
import argparse
import numpy as np
import pickle
import matplotlib

matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt
import concurrent.futures

# -----------------------------------------------------------------------------
# 1) Parse CLI once, define all globals
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Parallel plotting of conformal intervals per trajectory"
)
parser.add_argument(
    "file_path",
    help="Base path for your data (e.g.   data/rico_test →"
    "loads data/rico_test_processed.npy and data/rico_test/cp_<method>.pkl)",
)
parser.add_argument(
    "-m",
    "--method",
    required=True,
    help="The cp method key (used to open  …/cp_<method>.pkl and to name the output dir)",
)
args = parser.parse_args()

# derive file‐names + output folder
base_path = args.file_path
method = args.method
processed_npy = f"{base_path}_processed.npy"
cp_pkl = os.path.join(base_path, f"cp_{method}.pkl")
out_dir = os.path.join(base_path, "cpSINDy", method)
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 2) Load the big arrays (once, at import time)
#    - `mmap_mode='r'` lets each worker only open a memmap, no full copy
# -----------------------------------------------------------------------------
print("Loading time series memmap …")
dset = np.load(processed_npy, mmap_mode="r")
times = dset[..., 0]  # shape = (n_traj, n_steps)
var_raw = dset[..., 1:]  # shape = (n_traj, n_steps, dim)
means = var_raw.mean(axis=(0, 1))

print("Loading conformal predictions outputs …")
with open(cp_pkl, "rb") as f:
    alphas, idx_test, lower, upper, rep = pickle.load(f)
# denormalize conformal predictions
lower *= means[None, None, None, :]
upper *= means[None, None, None, :]
rep *= means[None, None, :]

# precompute colors
cmap = plt.get_cmap("autumn_r")
colors = cmap(np.tanh(np.pi * np.array(alphas)))


# -----------------------------------------------------------------------------
# 3) Worker function: plot a single trajectory `j`
# -----------------------------------------------------------------------------
def plot_trajectory(j):
    """
    Draw the two-subplot figure for trajectory j,
    fill_between using lower/upper[:,j,...], then save PDF.
    """
    t = times[idx_test[j]]
    true = var_raw[idx_test[j]]
    rep_traj = rep[j]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- subplot q_c (index 0) ---
    axs[0].plot(t, true[:, 0], "b-o", ms=0.5, label=r"Exact, $q_c$")
    for k, alpha in enumerate(alphas):
        axs[0].fill_between(
            t[1:],
            lower[k, j, :, 0],
            upper[k, j, :, 0],
            color=colors[k],
            alpha=0.3,
            label=f"{100*(1-alpha):.0f}% coverage",
        )
    axs[0].plot(t[1:], rep_traj[:, 0], "r-x", ms=0.5, label=r"SINDy (repr.), $q_c$")
    axs[0].set_xlabel("t [seconds]")
    axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[0].grid(True)

    # --- subplot q_r (index 1) ---
    axs[1].plot(t, true[:, 1], "b-o", ms=0.5, label=r"Exact, $q_r$")
    for k, alpha in enumerate(alphas):
        axs[1].fill_between(
            t[1:],
            lower[k, j, :, 1],
            upper[k, j, :, 1],
            color=colors[k],
            alpha=0.3,
            label=f"{100*(1-alpha):.0f}% coverage",
        )
    axs[1].plot(t[1:], rep_traj[:, 1], "r-x", ms=0.5, label=r"SINDy (repr.), $q_r$")
    axs[1].set_xlabel("t [seconds]")
    axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[1].grid(True)

    fig.suptitle(f"Trajectory No. {idx_test[j]}, {method}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_file = os.path.join(out_dir, f"intervals_{idx_test[j]}.pdf")
    fig.savefig(out_file)
    plt.close(fig)

    return idx_test[j]  # so we can report progress


# -----------------------------------------------------------------------------
# 4) Main: spawn a ProcessPool and map over all trajectories
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    n_traj = rep.shape[0]
    print(
        f"Launching parallel plot of {n_traj} trajectories "
        f"using {os.cpu_count()} processes…"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # executor.map will feed each j into plot_trajectory(j)
        for j in executor.map(plot_trajectory, range(n_traj)):
            print(f"Finished trajectory {j}")

    print("All plots done.")
