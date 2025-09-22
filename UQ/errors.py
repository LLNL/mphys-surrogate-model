"""
Script plots areas between prediction/confidence bands as a function of time on the AE-X architecture 
for different subsets of the network. Same as errors.py, except plots results from all models at once.
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.append(parent_directory)  # parent_directory = ~/mphys-surrogate-model

from src import data_utils as du

params = {
    "random_seed": 1952,
    "latent_dim": 3,
}

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-s",
    "--subset",
    type=str,
    required=True,
    choices=["nomass", "all"],
    help="which subset of the network you want to plot conformal predictions on:"
    "-nomass: plot all but the mass"
    "-all: plot all, including the mass",
)
parser.add_argument(
    "-p",
    "--p",
    type=int,
    default=20,
    help="The p indicates what *percent* you want to dedicate out of the full data for calibration."
    "Default is 20%, in which case p=20.",
)
parser.add_argument(
    "-u",
    "--uncertainty",
    type=str,
    default="conformal",
    choices=["conformal"],
    help="Types of uncertainty intervals to plot."
    "Default (and only one implemented currently) is conformal.",
)
args = parser.parse_args()

calib_size = args.p

DSD_areas = {}
m_diffs = {}
volumes = {}

import numpy as np
import math


# calculate volumes of ellipse
def ellipse_volumes_pd(Sigma_inv, taus):
    """
    Sigma_inv: (m, d, d) array of PD matrices A
    taus:      (len(alphas), m)     nonnegative radii t
    returns:   (len(alphas), m)     volumes  V_d * t^{d/2} / sqrt(det A)
    """
    m, d, d2 = Sigma_inv.shape
    assert d == d2, "Sigma_inv must be (m,d,d)"
    assert taus.shape[1] == m, "taus must be (k,m)"

    # Unit d-ball volume
    Vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    # Stable log-determinants; sign should be +1 for PD matrices
    sign, logdet = np.linalg.slogdet(Sigma_inv)  # (m,)
    if not np.all(sign > 0):
        raise ValueError("All A must be positive definite (got non-positive det).")

    inv_sqrt_detA = np.exp(-0.5 * logdet)  # (m,)

    # Broadcast over k for taus
    return Vd * (taus ** (d / 2)) * inv_sqrt_detA  # (k,m)


models = ["SINDy", "NNdzdt", "AR"]
for model in models:
    # load results from conformal predictions
    cp_results_file = (
        os.path.basename(os.path.normpath(args.data_name))
        + "_split"
        + str(calib_size)
        + ".pkl"
    )
    pickle_path = os.path.join(
        parent_directory,
        "UQ",
        args.uncertainty,
        "results",
        "ae_" + model,
        cp_results_file,
    )
    with open(pickle_path, "rb") as f:
        alphas, _, DSD_bands, m_bands, latent_dict = pickle.load(f)

    """
    First dim for these two below corresponds to architecture (in this order):
    -decoder, full (end-to-end) 
    Then, the dimensions are the same for both DSD_* and m_*: (# alphas, # samples, # times, # dims)
    """
    DSD_lower = DSD_bands[0]
    DSD_upper = DSD_bands[1]
    # the difference between bands should be the same across all samples...we take the mean in case this is not the case
    DSD_diff = [np.mean(DSD_upper[arch] - DSD_lower[arch], axis=1) for arch in range(2)]

    m_lower = m_bands[0]
    m_upper = m_bands[1]
    # the difference between bands should be the same across all samples...we take the mean in case this is not the case
    m_diff = np.mean(m_upper - m_lower, axis=1)

    # load data
    outputs = du.open_mass_dataset(
        name=args.data_name,
        data_dir=Path(parent_directory) / "data",
        sample_time=None,
        test_size=1 - 0.01 * calib_size,
        random_state=params["random_seed"],
    )

    # compute areas between DSD prediction bands
    rbins_mid = (
        np.log(outputs["r_bins_edges"]) + np.log(outputs["r_bins_edges_r"])
    ) / 2  # midpoints of bins
    # size of domain to normalize integrals
    domain_size = np.log(outputs["r_bins_edges_r"][-1]) - np.log(
        outputs["r_bins_edges"][0]
    )
    # composite trapezoidal rule for DSD outputs; average across latent space coordinates
    DSD_areas[model] = [
        np.trapz(DSD_diff[0], x=rbins_mid, axis=-1) / domain_size,
        np.trapz(DSD_diff[1], x=rbins_mid, axis=-1) / domain_size,
    ]
    m_diffs[model] = m_diff

    # finally, compute the volumes of the conformal ellipses
    Sigma_inv = latent_dict["Sigma_inv"]
    taus = latent_dict["taus"]
    volumes[model] = ellipse_volumes_pd(
        Sigma_inv=Sigma_inv, taus=taus
    )  # array of ellipse volumes


# define figures. Columns correspond to parts of architecture
if args.subset == "nomass":
    arch_labels = ["reconstruction", "latent dynamics", "end-to-end"]
elif args.subset == "all":
    arch_labels = ["reconstruction", "latent dynamics", "mass", "end-to-end"]
arch_to_idx = {
    "reconstruction": 0,
    "end-to-end": 1,
}

n_cols = len(arch_labels)
fig, ax = plt.subplots(
    ncols=n_cols,
    figsize=(2.5 * n_cols, n_cols),
    sharey=False,
    constrained_layout=True,
)

# grab colors from matplotlib default cycle
model_colors = {
    "Data": "tab:blue",
    "SINDy": "tab:orange",
    "NNdzdt": "tab:green",
    "AR": "tab:red",
}

# linestyles for miscoverage levels
alpha_linestyles = {
    alphas[0]: "solid",
    alphas[1]: "dashed",
    alphas[2]: "dashdot",
    alphas[3]: "dotted",
}

for j, label in enumerate(arch_labels):  # loop through architectures (columns)
    for model in models:  # loop through models
        for k, alpha in enumerate(alphas):  # loop through alpha values
            if label == "mass":
                ax[j].plot(
                    outputs["dsd_time"],
                    m_diffs[model][k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                    label=r"$\alpha={}$%, {}".format(100 * alpha, model),
                )
            elif label == "latent dynamics":
                ax[j].plot(
                    outputs["dsd_time"],
                    volumes[model][k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                )
            else:
                data_idx = arch_to_idx[label]
                ax[j].plot(
                    outputs["dsd_time"],
                    DSD_areas[model][data_idx][k],
                    color=model_colors[model],
                    linestyle=alpha_linestyles[alpha],
                )

    ax[j].set_xlabel("time [s]")

    # y-labels and limits
    if label == "mass":
        ax[j].set_ylabel("mass [-]")
    else:
        ax[j].set_ylabel(f"{label} [-]")

# legend for alpha values (colors = models)
color_handles = [Line2D([0], [0], color=model_colors[model], lw=2) for model in models]
legend1 = ax[-1].legend(
    color_handles,
    models,
    bbox_to_anchor=(1.05, 1),
    title="Dynamics model",
    loc="upper left",
)

# legend for miscoverage (linestyles)
marker_handles = [
    Line2D([0], [0], color="black", linestyle=alpha_linestyles[alpha])
    for alpha in alphas
]
legend2 = ax[-1].legend(
    marker_handles,
    [r"$\alpha={}$%".format(100 * alpha) for alpha in alphas],
    bbox_to_anchor=(1.05, 0),
    title="Miscoverage rate",
    loc="lower left",
)
plt.gca().add_artist(legend1)

fig.savefig(
    os.path.join(
        parent_directory,
        "results",
        "UQ",
        args.uncertainty,
        f"errors_{os.path.basename(os.path.normpath(args.data_name))}_{args.subset}.pdf",
    ),
    bbox_inches="tight",
)
