# comparing results of SINDy with the data for the Kessler model
# one input to be piped in: which trajectory_no to test

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt

# specify trajectory number to test
traj_no = int(input())

dset = np.load("data/box64_processed.npy")  # load original data

# get time and state data
times = dset[:, :, 0]
states = dset[:, :, 1:]

# plot original trajectories
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# First subplot: q_c
axs[0].plot(
    times[traj_no],
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
    times[traj_no],
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
plt.savefig("trajectory.pdf")
plt.close()
