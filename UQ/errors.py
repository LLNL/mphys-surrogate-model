"""
Script plots areas between prediction/confidence bands as a function of time on the AE-X architecture 
for different subsets of the network.
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    "-a",
    "--model",
    type=str,
    required=True,
    choices=["AR", "NNdzdt", "SINDy"],
    help="the dynamic model/architecture used (required): AR, NNdzdt, or SINDy",
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

# load results from conformal predictions
cp_results_file = args.data_name + "_" + args.method + ".pkl"
pickle_path = os.path.join(
    parent_directory,
    "UQ",
    args.uncertainty,
    "results",
    "ae_" + args.model,
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
if calib_size:
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
domain_size = np.log(outputs["r_bins_edges_r"][-1]) - np.log(outputs["r_bins_edges"][0])
# composite trapezoidal rule for DSD outputs; average across latent space coordinates
DSD_areas = [
    np.trapz(DSD_diff[0], x=rbins_mid, axis=-1) / domain_size,
    np.mean(DSD_diff[1], axis=-1),
    np.trapz(DSD_diff[2], x=rbins_mid, axis=-1) / domain_size,
]

# define figures. Columns correspond to parts of architecture
if args.subset == "nomass":
    n_cols = 3
elif args.subset == "all":
    n_cols = 4
(fig, ax) = plt.subplots(
    ncols=n_cols,
    figsize=(2.5 * n_cols, 2),
    sharey=False,
    constrained_layout=True,
)
arch_labels = ["Reconstruction", "Latent dynamics", "End-to-end", "Mass"]
for j in range(n_cols):  # loop through architectures (columns)
    # loop through alpha values
    for i, alpha in enumerate(alphas):
        if j == 3:
            ax[j].plot(
                outputs["dsd_time"],
                m_diff[i],
                label=r"$\alpha={}$%".format(100 * alpha),
            )
        else:
            ax[j].plot(
                outputs["dsd_time"],
                DSD_areas[j][i],
                label=r"$\alpha={}$%".format(100 * alpha),
            )
    ax[j].set_title(arch_labels[j])
    ax[j].set_xlabel("time [s]")
    if j == 3:
        ax[j].set_ylim(-0.1 * m_diff.max(), 1.1 * m_diff.max())
        ax[j].set_ylabel("normalized mass \n [-]")
    else:
        ax[j].set_ylim(-0.1 * DSD_areas[j].max(), 1.1 * DSD_areas[j].max())
        if j == 1:
            ax[j].set_ylabel("[-]")
        else:
            ax[j].set_ylabel("mass mixing ratio \n [kg liquid/kg air]")
    ax[j].locator_params(axis="x", nbins=4)  # Aim for 4 major ticks on the x-axis
    ax[j].locator_params(axis="y", nbins=3)  # Aim for 3 major ticks on the y-axis
ax[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig.savefig(
    os.path.join(
        parent_directory,
        "results",
        "UQ",
        args.uncertainty,
        "ae_" + args.model,
        f"errors_{args.data_name}_{args.method}_{args.subset}.pdf",
    ),
    bbox_inches="tight",
)
