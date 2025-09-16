"""
Script plots areas between prediction/confidence bands as a function of time on the AE-X architecture 
for different subsets of the network. Same as errors.py, except plots results from all CP methods at once.
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
    "-m",
    "--method",
    type=str,
    required=True,
    help="which conformal predictions to plot (required): full, split[p], or cv+[k]",
)
parser.add_argument(
    "-u",
    "--uncertainty",
    type=str,
    default="conformal",
    choices=["conformal", "ensemble"],
    help="whether you would like to plot intervals from conformal or ensemble/bootsrapped predictions. Default is conformal.",
)
args = parser.parse_args()

method = args.method
calib_size = None
if method[:5] == "split":
    calib_size = float(method[5:])
    method = "split"
if method[:3] == "cv+":
    method = "cv+"
if method not in [
    "split",
    "full",
    "cv+",
]:  # raise error if method is not one of the list above
    raise ValueError("Conformal predictions method specified has not been implemented.")

calib_size = None
calib_size = 20
k = 20

DSD_areas = {}
m_diffs = {}

models = ["SINDy", "NNdzdt", "AR"]
for model in models:
    # load results from conformal predictions
    if method == "vanilla":
        arg_method = "full"
    if method == "split":
        arg_method = "split" + str(calib_size)
    if method == "cv+":
        arg_method = "cv+" + str(k)
    cp_results_file = (
        os.path.basename(os.path.normpath(args.data_name)) + "_" + arg_method + ".pkl"
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
        alphas, test_size, _, DSD_bands, m_bands = pickle.load(f)

    """
    First dim for these three below corresponds to architecture (in this order):
    -decoder, latent, full (end-to-end) 
    Then, the dimensions are the same for both DSD_* and m_*: (# alphas, # samples, # times, # dims)
    """
    DSD_lower = DSD_bands[0]
    DSD_upper = DSD_bands[1]
    # the difference between bands should be the same across all samples...we take the mean in case this is not the case
    DSD_diff = [np.mean(DSD_upper[arch] - DSD_lower[arch], axis=1) for arch in range(3)]

    m_lower = m_bands[0]
    m_upper = m_bands[1]
    # the difference between bands should be the same across all samples...we take the mean in case this is not the case
    m_diff = np.mean(m_upper - m_lower, axis=1)

    # load data
    if method == "split":
        outputs = du.open_mass_dataset(
            name=args.data_name,
            data_dir=Path(parent_directory) / "data",
            sample_time=None,
            test_size=test_size,
            calib_size=0.01 * calib_size,
            random_state=params["random_seed"],
        )
    else:
        outputs = du.open_mass_dataset(
            name=args.data_name,
            data_dir=Path(parent_directory) / "data",
            sample_time=None,
            test_size=test_size,
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
        np.mean(DSD_diff[1], axis=-1),
        np.trapz(DSD_diff[2], x=rbins_mid, axis=-1) / domain_size,
    ]
    m_diffs[model] = m_diff

# define figures. Columns correspond to parts of architecture
if args.subset == "nomass":
    arch_labels = ["reconstruction", "latent dynamics", "end-to-end"]
    arch_to_idx = {
        "reconstruction": 0,
        "latent dynamics": 1,
        "end-to-end": 2,
    }
elif args.subset == "all":
    arch_labels = ["reconstruction", "latent dynamics", "mass", "end-to-end"]
    arch_to_idx = {
        "reconstruction": 0,
        "latent dynamics": 1,
        "end-to-end": 2,
        # no entry for "mass" because it's handled by m_diffs
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
        data_idx = arch_to_idx[label]
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
        f"errors_{os.path.basename(os.path.normpath(args.data_name))}_all_{args.subset}.pdf",
    ),
    bbox_inches="tight",
)
