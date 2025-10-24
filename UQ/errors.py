"""
Script plots areas between prediction/confidence bands as a function of time on the AE-X architecture
for different subsets of the network. This version is optimized and switches layout:
  - subset=all    -> 2x2 grid: [reconstruction, end-to-end; latent, mass]
  - subset=nomass -> 3x1 stack: reconstruction, latent, end-to-end
"""
import os
import sys
import math
import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)  # parent_directory = ~/mphys-surrogate-model

from src import data_utils as du

# ----------------
# Args & params
# ----------------
params = {"random_seed": 1952, "latent_dim": 3}

parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-s",
    "--subset",
    type=str,
    required=True,
    choices=["nomass", "all"],
    help="which subset to plot: 'nomass' = reconstruction, latent, end-to-end; 'all' = add mass too",
)
parser.add_argument(
    "-p",
    "--p",
    type=int,
    default=20,
    help="Percent of data used for calibration (default 20).",
)
parser.add_argument(
    "-u",
    "--uncertainty",
    type=str,
    default="conformal",
    choices=["conformal"],
    help="Type of uncertainty intervals to plot (currently only 'conformal').",
)
args = parser.parse_args()
calib_size = args.p


# ----------------
# Utilities
# ----------------
def ellipse_volumes_pd(Sigma_inv, taus):
    """
    Sigma_inv: (T, d, d) array of PD precision matrices A_t
    taus:      (K, T)     nonnegative radii (per alpha k and time t)
    returns:   (K, T)     volumes  V_d * tau^{d/2} / sqrt(det A_t)
    """
    T, d, d2 = Sigma_inv.shape
    assert d == d2, "Sigma_inv must be (T,d,d)"
    # Unit d-ball volume
    Vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    # Stable log-determinants
    # NOTE: np.linalg.slogdet returns (sign, logdet); det(A) = sign*exp(logdet) with sign=+1 for PD.
    _, logdet = np.linalg.slogdet(Sigma_inv)  # (T,)
    inv_sqrt_detA = np.exp(-0.5 * logdet)  # (T,)
    # Broadcast over K
    # taus: (K,T) ; inv_sqrt_detA: (T,) -> (K,T)
    return Vd * (taus ** (d / 2)) * inv_sqrt_detA[None, :]


# ----------------
# Load dataset once
# ----------------
outputs = du.open_mass_dataset(
    name=args.data_name,
    data_dir=Path(parent_directory) / "data",
    sample_time=None,
    test_size=1 - 0.01 * calib_size,
    random_state=params["random_seed"],
)
t_full = outputs["dsd_time"]  # (T,)
t_pos = t_full[1:]  # (T-1,) for latent volumes/mass (to match your original)
# Bin midpoints (log-space) and normalization length for DSD integrals
rbins_mid = (np.log(outputs["r_bins_edges"]) + np.log(outputs["r_bins_edges_r"])) / 2.0
domain_size = np.log(outputs["r_bins_edges_r"][-1]) - np.log(outputs["r_bins_edges"][0])

# ----------------
# Accumulators
# ----------------
models = ["SINDy", "NNdzdt", "AR"]
DSD_areas = {}  # model -> [areas_recon (K,T), areas_end2end (K,T)]
m_diffs = {}  # model -> (K,T)
volumes = {}  # model -> (K,T) latent ellipse volumes
alphas = np.array([])  # will take from first file

# ----------------
# Load all results once and compute metrics vectorized
# ----------------
for model in models:
    cp_results_file = (
        f"{os.path.basename(os.path.normpath(args.data_name))}_split{calib_size}.pkl"
    )
    pickle_path = os.path.join(
        parent_directory,
        "UQ",
        args.uncertainty,
        "results",
        f"ae_{model}",
        cp_results_file,
    )
    with open(pickle_path, "rb") as f:
        alphas_m, _, DSD_bands, m_bands, latent_dict = pickle.load(f)
    if alphas.size == 0:
        alphas = np.array(alphas_m, dtype=float)
    # DSD bands: list of two arrays [decoder, full], each (K, N, T, B)
    DSD_lower, DSD_upper = DSD_bands[0], DSD_bands[1]
    # Band widths, averaged over samples N and integrated over bins (vectorized)
    # widths: (K, N, T, B) -> mean over N -> (K, T, B) -> integrate over bins -> (K, T)
    dsd_widths = [DSD_upper[i] - DSD_lower[i] for i in range(2)]  # list of (K,N,T,B)
    dsd_widths_mean = [w.mean(axis=1) for w in dsd_widths]  # list of (K,T,B)
    areas = [
        np.trapz(wm, x=rbins_mid, axis=-1) / domain_size for wm in dsd_widths_mean
    ]  # [(K,T),(K,T)]
    DSD_areas[model] = areas  # [recon, end2end]

    # Mass band widths: (K,N,T) -> mean over N -> (K,T)
    m_lower, m_upper = m_bands[0], m_bands[1]
    m_diffs[model] = (m_upper - m_lower).mean(axis=1)  # (K,T)

    # Latent volumes from precision & taus (already per-time)
    Sigma_inv = latent_dict["Sigma_inv"]  # (T,d,d)
    taus = latent_dict["taus"]  # (K,T)
    volumes[model] = ellipse_volumes_pd(Sigma_inv, taus)  # (K,T)
    print(f"Done with {model}.")

# ----------------
# Plotting
# ----------------
# Architecture labels & mapping
if args.subset == "nomass":
    arch_labels = ["Reconstruction", "Latent dynamics", "End-to-end"]
else:  # "all"
    # 2x2 with recon / end-to-end top, latent / mass bottom
    arch_labels = ["Reconstruction", "End-to-end", "Latent dynamics", "Mass"]

# Common style
model_colors = {"SINDy": "tab:orange", "NNdzdt": "tab:green", "AR": "tab:red"}
linestyles = ["solid", "dashed", "dashdot", "dotted"]
alpha_linestyles = {
    alpha: linestyles[i % len(linestyles)] for i, alpha in enumerate(alphas)
}


def plot_panel(ax, label):
    """Plot one panel given the axis and which label it is."""
    for model in models:
        if label == "Mass":
            # (K,T) -> plot vs t_pos starting from index 1 to match previous behavior
            y = m_diffs[model][:, 1:]
            for k, alpha in enumerate(alphas):
                ax.plot(
                    t_pos,
                    y[k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                )
            ax.set_yscale("log")
            ax.set_ylabel("Mass interval")
        elif label == "Latent dynamics":
            y = volumes[model][:, 1:]  # (K,T-1)
            for k, alpha in enumerate(alphas):
                ax.plot(
                    t_pos,
                    y[k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                )
            ax.set_yscale("log")
            ax.set_ylabel("Latent volume interval")
        else:
            # reconstruction or end-to-end
            idx = 0 if label == "Reconstruction" else 1
            y = DSD_areas[model][idx]  # (K,T)
            for k, alpha in enumerate(alphas):
                ax.plot(
                    t_full,
                    y[k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                )
            ax.set_ylabel(f"{label} interval")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_title(label)


# Layout logic
if args.subset == "all":
    fig, axes = plt.subplots(
        2, 2, figsize=(6, 5), sharex=False, sharey=False, constrained_layout=True
    )
    order = ["Reconstruction", "End-to-end", "Latent dynamics", "Mass"]  # row-major
    for ax, label in zip(axes.flat, order):
        plot_panel(ax, label)
else:  # nomass -> 3x1 vertical
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.6), sharex=False, sharey=False, constrained_layout=True
    )
    for ax, label in zip(axes, ["Reconstruction", "Latent dynamics", "End-to-end"]):
        plot_panel(ax, label)

# Legends: (1) models (colors) and (2) miscoverage (linestyles)
color_handles = [Line2D([0], [0], color=model_colors[m], lw=2) for m in models]
ls_handles = [
    Line2D([0], [0], color="black", linestyle=alpha_linestyles[a]) for a in alphas
]

# place legends on the figure level to avoid crowding
fig.legend(
    color_handles,
    models,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    title="Dynamics model",
)
fig.legend(
    ls_handles,
    [rf"$\alpha={100*a:.0f}\%$" for a in alphas],
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0),
    title="Miscoverage rate",
)

# Save
out_path = os.path.join(
    parent_directory,
    "figures",
    "UQ",
    f"errors_{os.path.basename(os.path.normpath(args.data_name))}_{args.subset}.pdf",
)
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
