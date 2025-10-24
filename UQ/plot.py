"""
Script plots conformal prediction results on the AE-X architecture at specified times, gridboxes/samples, and subsets of the network.
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
    choices=["decoder", "full", "mass"],
    help="which subset of the network you want to plot conformal predictions on: decoder, full, or mass"
    "-decoder: the reconstruction/autoencoder only"
    "-full: the entire network architecture (reconstruction+dynamics)"
    "-mass: the (normalized) mass as a function of time"
    "(sorry, 3D latent trajectory plots not implemented yet!)",
)
parser.add_argument(
    "-t",
    "--tplt",
    type=str,
    required=True,
    help="the indices of which times to plot (required), space separated, inputed as a string",
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
parser.add_argument(
    "-g",
    "--id",
    type=int,
    required=True,
    help="the index of which samples/gridboxes to plot (required), space separated."
    "indices are respect to the full dataset, NOT with respect to the testing data",
)
parser.add_argument(
    "-title",
    "--title",
    type=str,
    default="y",
    choices=["y", "n"],
    help="Do you want an overall title (y/n)?",
)
args = parser.parse_args()

calib_size = args.p

models = ["SINDy", "NNdzdt", "AR"]

# load results from conformal predictions
cp_results_file = (
    os.path.basename(os.path.normpath(args.data_name))
    + "_split"
    + str(calib_size)
    + ".pkl"
)

id = args.id  # a single int

lowers = {}
uppers = {}
reps = {}

for model in models:
    pickle_path = os.path.join(
        parent_directory,
        "UQ",
        args.uncertainty,
        "results",
        "ae_" + model,
        cp_results_file,
    )
    with open(pickle_path, "rb") as f:
        alphas, _, DSD_bands, m_bands, _ = pickle.load(f)

    # load data
    outputs = du.open_mass_dataset(
        name=args.data_name,
        data_dir=Path(parent_directory) / "data",
        sample_time=None,
        test_size=1 - 0.01 * calib_size,
        random_state=params["random_seed"],
    )

    """
    Get indices to plot relative to the test data. 
    If any are not in there, then return error.
    """
    pos_map = {value: idx for idx, value in enumerate(outputs["idx_test"])}

    if id not in pos_map:
        raise KeyError(f"Id {id} is not in the testing data")

    # if present, then return their positions
    ids_rel_to_test = pos_map[id]
    tplt = np.fromstring(args.tplt, dtype=int, sep=" ")
    subset = args.subset

    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS.values())

    if subset == "decoder":
        lowers[model] = DSD_bands[0][0]
        uppers[model] = DSD_bands[1][0]
        reps[model] = DSD_bands[2][0]
    elif subset == "full":
        lowers[model] = DSD_bands[0][1]
        uppers[model] = DSD_bands[1][1]
        reps[model] = DSD_bands[2][1]
    elif subset == "mass":
        lowers[model] = m_bands[0]
        uppers[model] = m_bands[1]
        reps[model] = m_bands[2]
    else:
        raise KeyError(
            "Subset of architecture indicated (via -s) has not been implemented"
        )
    print(f"Done with {model}.")

if subset == "mass":
    (fig, ax) = plt.subplots(
        ncols=len(models),
        nrows=1,
        figsize=(2.5 * len(models), 2 * 3),
        sharey=True,
    )
    for i, m in enumerate(models):
        for k_alpha, alpha in enumerate(
            alphas
        ):  # ensure masses are always non-negative
            ax[i].fill_between(
                outputs["dsd_time"],
                np.maximum(0, lowers[m][k_alpha, ids_rel_to_test]),
                uppers[m][k_alpha, ids_rel_to_test],
                color=colors[k_alpha],
                alpha=0.4,
                label=f"{100*(1-alpha)}% coverage",
            )
        ax[i].plot(
            outputs["dsd_time"],
            outputs["m_test"][ids_rel_to_test],
            label="Data",
            color="black",
            linestyle="solid",
            linewidth=1.5,
        )
        ax[i].plot(
            outputs["dsd_time"],
            reps[m][ids_rel_to_test],
            label="Model",
            color="black",
            linestyle="dashed",
            linewidth=1.5,
        )  # representative band
        ax[i].set_title(f"AE-{m}")
        ax[i].set_xlabel("Elapsed time (s)")
    ax[0].set_ylabel("Normalized mass [-]")
    ax[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    bin_mids = 0.5 * (outputs["r_bins_edges"] + outputs["r_bins_edges_r"])
    (fig, ax) = plt.subplots(
        ncols=len(models),
        nrows=len(tplt),
        figsize=(2.5 * len(models), 2 * len(tplt)),
        sharey=True,
    )
    for i, m in enumerate(models):
        for j, t in enumerate(tplt):
            for k_alpha, alpha in zip(reversed(range(len(alphas))), reversed(alphas)):
                ax[j][i].fill_between(
                    outputs["r_bins_edges"],
                    np.maximum(0, lowers[m][k_alpha, ids_rel_to_test, t]),
                    uppers[m][k_alpha, ids_rel_to_test, t],
                    step="pre",
                    color=colors[k_alpha],
                    alpha=0.4,
                    label=f"{100*(1-alpha)}% coverage",
                )
            ax[j][i].step(
                bin_mids,
                outputs["x_test"][ids_rel_to_test, t],
                label="Data",
                color="black",
                linestyle="solid",
                linewidth=1.5,
            )
            ax[j][i].step(
                bin_mids,
                reps[m][ids_rel_to_test, t],
                label="Model",
                color="gainsboro",
                linestyle="solid",
                linewidth=1.5,
            )  # representative band
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"AE-{m}")
        ax[-1][i].set_xlabel("radius (m)")
        ax[j][i].set_ylim(0, 0.8)
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(
            f'dmdlnr at t={outputs["dsd_time"][t]} s'
        )
    ax[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
if args.title == "y":
    fig.suptitle(
        f"Conformal predictions, {subset} network",
        fontsize=14,
        # y=1.05
    )
# fig.subplots_adjust(hspace=0.5)

fig.savefig(
    os.path.join(
        parent_directory,
        "figures",
        "UQ",
        f"{os.path.basename(os.path.normpath(args.data_name))}_{subset}_{id}.pdf",
    ),
    bbox_inches="tight",
)
fig.clf()