# comparing results of SINDy with the data for the Kessler model
# one input to be piped in: which trajectory_no to test

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import dill

# specify trajectory number to test
traj_no = int(input())

# load trained PySINDy model
with open("sindy_model.pkl", "rb") as f:
    model = dill.load(f)
file_path = "data/box64_processed.npy"
dset = np.load(file_path)  # load original data

# get time and state data
times = dset[:, :, 0]
states = dset[:, :, 1:]
var_means = states.mean(axis=(0, 1))
states = states / var_means[None, None, :]  # normalize each variable by its own mean


# vector field from SINDy
def fun(t, x):
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    dxdt = model.predict(x.reshape(1, -1))
    return dxdt[0]


# epsilon scaling for fifth-order damping term
# ratio of this term to the smallest yet non-zero coefficient
eps = 0.01 * 10 ** (-5)
abs_coefs = np.abs(model.coefficients())
final_eps = eps * np.min(abs_coefs[abs_coefs > 0]).flatten()  # actual coefficient value


# damped version of vector field from SINDy
def fun_damped(t, x):
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    x_eval = x.reshape(1, -1)
    dxdt = model.predict(x_eval) - final_eps * (x_eval**5)
    return dxdt[0]


t = times[traj_no]
sol_y = np.zeros((len(t), 2))

try:
    # solve SINDy model using initial condition from dataset for trajectory indicated in traj_no
    # loop through each time in t, do this to ensure that solution stays non-negative
    for i, ti in enumerate(t):
        if i == 0:  # save initial condition first
            sol_y[i] = states[traj_no][i]
        else:
            # integrate from t[i-1] to t[i]
            sol = solve_ivp(
                fun, (t[i - 1], ti), sol_y[i - 1], method="LSODA", t_eval=[ti]
            )
            # print('Solver status:', sol.status) # print termination condition
            if sol.status == -1:
                raise Exception(
                    "Integration failed, trying damped version."
                )  # raise a custom exception for LSODA
            sol_y[i] = sol.y.flatten()
            # make solution non-negative
            sol_y[i] = np.maximum(sol_y[i], 0)

    # denormalize everything
    states_j = states[traj_no] * var_means[None, :]
    sol_y *= var_means[None, :]

    # plot original trajectories and model interpolation
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # First subplot: q_c
    axs[0].plot(
        t, states_j[:, 0], color="blue", marker="o", markersize=3, label=r"Exact, $q_c$"
    )
    axs[0].plot(
        t,
        sol_y[:, 0],
        color="orange",
        marker="x",
        markersize=3,
        label=r"SINDy (non-damped), $q_c$",
    )
    axs[0].set_xlabel("t [seconds]")
    axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[0].grid(True)

    # Second subplot: q_r
    axs[1].plot(
        t, states_j[:, 1], color="blue", marker="o", markersize=3, label=r"Exact, $q_r$"
    )
    axs[1].plot(
        t,
        sol_y[:, 1],
        color="orange",
        marker="x",
        markersize=3,
        label=r"SINDy (non-damped), $q_r$",
    )
    axs[1].set_xlabel("t [seconds]")
    axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[1].grid(True)

    fig.suptitle(f"Trajectory No. {traj_no}")
    plt.tight_layout()
    plt.savefig(
        file_path.removesuffix("_processed.npy")
        + "/SINDy/trajectory"
        + str(traj_no)
        + ".pdf"
    )
    print("Finished trajectory no.", traj_no, "(non-damped).")
except:  # if original system fails, try to integrate damped version
    try:
        # solve SINDy model using initial condition from dataset for trajectory indicated in traj_no
        # loop through each time in t, do this to ensure that solution stays non-negative
        for i, ti in enumerate(t):
            if i == 0:  # save initial condition first
                sol_y[i] = states[traj_no][i]
            else:
                # integrate from t[i-1] to t[i]
                sol = solve_ivp(
                    fun_damped,
                    (t[i - 1], ti),
                    sol_y[i - 1],
                    method="LSODA",
                    t_eval=[ti],
                )
                # print('Solver status:', sol.status) # print termination condition
                if sol.status == -1:
                    raise Exception(
                        "Damped integration failed."
                    )  # raise a custom exception for LSODA
                sol_y[i] = sol.y.flatten()
                # make solution non-negative
                sol_y[i] = np.maximum(sol_y[i], 0)

        # denormalize everything
        states_j = states[traj_no] * var_means[None, :]
        sol_y *= var_means[None, :]

        # plot original trajectories and model interpolation
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # First subplot: q_c
        axs[0].plot(
            t,
            states_j[:, 0],
            color="blue",
            marker="o",
            markersize=3,
            label=r"Exact, $q_c$",
        )
        axs[0].plot(
            t,
            sol_y[:, 0],
            color="orange",
            marker="x",
            markersize=3,
            label=r"SINDy (damped), $q_c$",
        )
        axs[0].set_xlabel("t [seconds]")
        axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[0].grid(True)

        # Second subplot: q_r
        axs[1].plot(
            t,
            states_j[:, 1],
            color="blue",
            marker="o",
            markersize=3,
            label=r"Exact, $q_r$",
        )
        axs[1].plot(
            t,
            sol_y[:, 1],
            color="orange",
            marker="x",
            markersize=3,
            label=r"SINDy (damped), $q_r$",
        )
        axs[1].set_xlabel("t [seconds]")
        axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[1].grid(True)

        fig.suptitle(f"Trajectory No. {traj_no}")
        plt.tight_layout()
        plt.savefig(
            file_path.removesuffix("_processed.npy")
            + "/SINDy/trajectory"
            + str(traj_no)
            + ".pdf"
        )
        print("Finished trajectory no.", traj_no, "(damped).")
    except:
        print("Unable to integrate trajectory no.", traj_no)
