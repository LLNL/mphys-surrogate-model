# runs E-SINDy and computes each trajectory rather than coefficients
# then, script will plot provided percentiles for the collection of trajectories

import numpy as np
import pysindy as ps
import argparse

parser = argparse.ArgumentParser(
    description="Run E-SINDy plotting, serially"
)

parser.add_argument("file_path",
    help="Path to your processed .npy file")

parser.add_argument("-B", "--boots",
    dest="B",
    type=int,
    required=True,
    help="Number of bootstrap replicates")

parser.add_argument("-t", "--traj_nos",
    dest="traj_nos",
    nargs="+",
    type=int,
    required=True,
    help="List of trajectory indices")

args = parser.parse_args()

# now variables are
file_path = args.file_path      # str
B         = args.B              # int
traj_nos  = args.traj_nos       # List[int]
    
# which quantiles do I want? (preset)
quantile_query = 0.01*np.array([1,2.5,5,50,95,97.5,99])

# get list of trajectories for times and states
dset = np.load(file_path)
times  = dset[..., 0]
states = dset[..., 1:]
var_means   = states.mean(axis=(0,1))            
states      = states / var_means[None,None,:] # normalize each variable by its own mean

from pysindy.optimizers import STLSQ

# a = 0.001 # threshold for autoconversion in kg water/kg air
# a *= 0.001225 / mean # convert tolerance to volume water/volume air, normalized 
# print(a)
# a = 2.340384636957035

# create custom fractional exponent library
library_functions = []
library_function_names = []
library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.875) * np.sign(y)) # for rate of accretion
library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.875)+')')
library_functions.append(lambda x, y: y * np.power(np.abs(x), 0.875) * np.sign(x)) # for rate of accretion
library_function_names.append(lambda x, y: y + ' ' + x+'^('+str(0.875)+')')
# turn these off below if you are doing dealing with a no-advection model
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.525) * np.sign(y)) # for rate of evaporation
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.525)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.7296) * np.sign(y)) # for ventilation factor term
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.7296)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.1346) * np.sign(y)) # for terminal fall velocity
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.1346)+')')
library_functions.append(lambda x: np.maximum(0, x - 2.340384636957035)) # autoconversion rate
library_function_names.append(lambda x: 'max(0,'+x+'-a)')
custom_library = ps.CustomLibrary(library_functions=library_functions, 
                                  function_names=library_function_names)
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False) + custom_library
# feature_library += custom_library * custom_library

# defining training for SINDy model
# uses STLSQ by default
threshold = 0.0001 # sparsity parameter
optimizer = STLSQ(threshold=threshold)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

# compute derivatives separately for each initial condition
derivatives = []

for i in range(len(times)):
    dfd = ps.differentiation.FiniteDifference()
    deriv = dfd._differentiate(states[i], times[i])  # each is (n_timesteps, n_state_vars)
    derivatives.append(deriv)

# now flatten
derivatives = np.concatenate(derivatives, axis=0)
flat_states = np.concatenate(states, axis=0)

traj_nos = np.array(traj_nos)
times = times[traj_nos] # time array for desired trajectories
states = states[traj_nos] # state array for desired trajectories

# import fast t-digest implementation
from fastdigest import TDigest

M = len(traj_nos)
N = flat_states.shape[1] # number of variables
Q = len(quantile_query) # number of quantiles to test

# preallocate array of quantiles:
# array of shape (desired trajectories, length of trajectory, variables, quantiles)
quantiles = np.empty(M, dtype=object)
# array of shape (desired trajectories, length of trajectory, variables)
trajectories = np.empty(M, dtype=object)
# preallocate quantiles and trajectory samples for each bootstrap: 
for j in range(M):
    len_traj = len(times[j])
    # preallocate quantile array
    quantiles[j] = np.empty((len_traj, N, Q), dtype=float)
    # make a J × N object array for trajectory m
    trajectory = np.empty((len_traj, N), dtype=object)
    for idx in np.ndindex(trajectory.shape):
        trajectory[idx] = TDigest() # initialize each element with TDigest()
    trajectories[j] = trajectory

# run loop for b(r)agging E-SINDy
fit_model = model.fit(flat_states, x_dot=derivatives) # run on whole data first to get # of output features and preallocate

# define integration functions
def fun(model, t, x):
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    dxdt = model.predict(x.reshape(1, -1)) 
    return dxdt[0]

def fun_damped(model, t, x):
    # epsilon scaling for fifth-order damping term
    # ratio of this term to the smallest yet non-zero coefficient 
    eps = 0.01 * 10 ** (-5)
    abs_coefs = np.abs(model.coefficients())
    final_eps = eps * np.min(abs_coefs[abs_coefs>0]).flatten() # actual coefficient value
    
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    x_eval = x.reshape(1, -1)
    dxdt = model.predict(x_eval) - final_eps * (x_eval ** 5)
    return dxdt[0]

from scipy.integrate import solve_ivp

# sample with replacement B times, update quantiles on trajectory values for given initial condition
# quantiles approximated in a streaming fashion using t-digests
for b in range(B): 
    indices = np.random.randint(0, len(flat_states), size=len(flat_states)) # indices to select for bootstrap
    model.fit(flat_states[indices], x_dot=derivatives[indices]) # solve model on resampled dataset 
    for j in range(M):
        t_j = times[j]
        N   = states[j].shape[1]
        y0  = states[j][0].copy()    # initial state for traj j

        # Try original RHS
        try:
            sol = solve_ivp(
                fun      = lambda t, y: fun(model, t, y),
                t_span   = (t_j[0], t_j[-1]),
                y0       = y0,
                method   = "BDF",
                t_eval   = t_j,
                positive = True,
            )
            if not sol.success:
                raise RuntimeError(f"Original BDF failed: {sol.message}")

        except (RuntimeError, ValueError) as e:
            # Fallback to damped RHS
            sol = solve_ivp(
                fun      = lambda t, y: fun_damped(model, t, y),
                t_span   = (t_j[0], t_j[-1]),
                y0       = y0,
                method   = "BDF",
                t_eval   = t_j,
                positive = True,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Trajectory {j}: both original and damped BDF failed: {sol.message}"
                ) from e

        # sol.y has shape (N, len(t_j)); transpose → (len(t_j), N)
        sol_y = sol.y.T

        # Feed the TDigest objects for trajectory j
        # (i,k) loops over time‐index and state‐index
        for (i, k), value in np.ndenumerate(sol_y):
            trajectories[j][i, k].update(value)
# we don't need the full derivatives or states anymore
del flat_states, derivatives

# denormalize everything for plotting
states *= var_means[None,None,:]
quantiles *= var_means[None,None,:,None]

# WARNING: the rest of the script is a lot less mutable than the parts above

import matplotlib
matplotlib.use("Agg")            # no GUI backend
import matplotlib.pyplot as plt

# loop through trajectories to compute quantiles, then make plots
for j in range(M):
    trajectory = trajectories[j]
    t = times[j]
    # compute desired quantiles for each time in trajectory
    for idx in np.ndindex(trajectory.shape):
        quantiles[j][idx] = np.array([trajectory[idx].quantile(q) for q in quantile_query])
    # plot quantiles and original trajectory
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # First subplot: q_c
    axs[0].plot(t, states[j][:, 0], color='blue', marker='o', markersize=0.5, label=r'Exact, $q_c$')
    axs[0].fill_between(t, quantiles[j][:, 0, 0], quantiles[j][:, 0, 6], 
                        color='yellow', alpha=0.3, label='98% CI')
    axs[0].fill_between(t, quantiles[j][:, 0, 1], quantiles[j][:, 0, 5], 
                        color='orange', alpha=0.3, label='95% CI')
    axs[0].fill_between(t, quantiles[j][:, 0, 2], quantiles[j][:, 0, 4], 
                        color='red', alpha=0.3, label='90% CI')
    axs[0].plot(t, quantiles[j][:, 0, 3], color='red', marker='x', 
                markersize=0.5, label=r'SINDy (median), $q_c$')
    axs[0].set_xlabel("t [seconds]")
    axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].grid(True)

    # Second subplot: q_r
    axs[1].plot(t, states[j][:, 1], color='blue', marker='o', markersize=0.5, label=r'Exact, $q_r$')
    axs[1].fill_between(t, quantiles[j][:, 1, 0], quantiles[j][:, 1, 6], 
                        color='yellow', alpha=0.3, label='98% CI')
    axs[1].fill_between(t, quantiles[j][:, 1, 1], quantiles[j][:, 1, 5], 
                        color='orange', alpha=0.3, label='95% CI')
    axs[1].fill_between(t, quantiles[j][:, 1, 2], quantiles[j][:, 1, 4], 
                        color='red', alpha=0.3, label='90% CI')
    axs[1].plot(t, quantiles[j][:, 1, 3], color='red', marker='x', 
                markersize=0.5, label=r'SINDy (median), $q_r$')
    axs[1].set_xlabel("t [seconds]")
    axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].grid(True)

    fig.suptitle(f"Trajectory No. {traj_nos[j]}")
    plt.tight_layout()
    plt.savefig(file_path.removesuffix("_processed.npy")+"/E-SINDy/CI/intervals"+str(traj_nos[j])+".pdf")
    plt.close()