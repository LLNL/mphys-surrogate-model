# comparing results of SINDy with the data for the Kessler model
# loop through all trajectories, save each to compare

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt

# load original data indicated by file path
file_path = input()
dset = np.load(file_path)

# get time and state data
times = dset[..., 0]
states = dset[..., 1:]

# loop through all trajectories and save plots for comparison
for traj_no in range(len(times)):
    t = times[traj_no]

    # plot original trajectories and model interpolation
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    try:
        # First subplot: q_c
        axs[0].plot(
            t,
            states[traj_no][:, 0],
            color="blue",
            marker="o",
            markersize=3,
            label=r"Exact, $q_c$",
        )
        axs[0].set_xlabel("t [seconds]")
        axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[0].grid(True)

        # Second subplot: q_r
        axs[1].plot(
            t,
            states[traj_no][:, 1],
            color="blue",
            marker="o",
            markersize=3,
            label=r"Exact, $q_r$",
        )
        axs[1].set_xlabel("t [seconds]")
        axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[1].grid(True)

        fig.suptitle(f"Trajectory No. {traj_no}")
        plt.tight_layout()
        plt.savefig(
            file_path.removesuffix("_processed.npy")
            + "/noSINDy/trajectory"
            + str(traj_no)
            + ".pdf"
        )
        print("Finished trajectory no.", traj_no)
    except:
        print("Unable to finish trajectory no.", traj_no)
    plt.close()
