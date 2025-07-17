# runs E-SINDy and computes each trajectory rather than coefficients
# then, script will plot provided percentiles for the collection of trajectories

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(
    description="Run plotting of conformal predictions, serially"
)

parser.add_argument("file_path",
    help="Name of data for cp results (e.g., rico_test)")

parser.add_argument("-m", "--method", type=str, required=True,
    help="Name of the cp method to plot")

args = parser.parse_args()
file_path = 'data/'+args.file_path
method = args.method

file_path_processed = file_path+'_processed.npy' 
cp_file_path = file_path+'/cp_'+method+'.pkl'      

print('loading data')
# load dataset
dset = np.load(file_path_processed)
times  = dset[..., 0]
states = dset[..., 1:]
var_means   = states.mean(axis=(0,1))            

# load all outputs from cp_SINDy.py/cp_SINDy_parallel.py
with open(cp_file_path, 'rb') as f:
    alphas, idx_test, lower, upper, rep = pickle.load(f)
# denormalize conformal predictions
lower *= var_means[None, None, None, :]
upper *= var_means[None, None, None, :]
rep *= var_means[None, None, :]

# isolate out all but the initial condition

# load plotting utilities and other things
import matplotlib
matplotlib.use("Agg")            # no GUI backend
import matplotlib.pyplot as plt

# use alphas to get color arguments for fill_between (the prediction intervals on the plot) 
cmap    = plt.get_cmap('autumn_r')     # 0→yellow, 1→red
colors  = cmap(np.tanh(np.pi*np.array(alphas)))            # take tanh to ensure that it is mostly red until very close to 0

print('starting plots')
# loop through trajectories to make plots
for j in range(rep.shape[0]):
    trajectory = rep[j]
    t = times[idx_test[j]]
    # compute desired quantiles for each time in trajectory
    # plot quantiles and original trajectory
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # First subplot: q_c
    axs[0].plot(t, states[idx_test[j], :, 0], color='blue', marker='o', markersize=0.5, label=r'Exact, $q_c$')
    for k_alpha, alpha in enumerate(alphas):
        axs[0].fill_between(t[1:], lower[k_alpha, j, :, 0], upper[k_alpha, j, :, 0], 
                            color=colors[k_alpha], alpha=0.3, label=f"{100*(1-alpha)}% coverage")
    axs[0].plot(t[1:], rep[j, :, 0], color='red', marker='x', 
                markersize=0.5, label=r'SINDy (representative trajectory), $q_c$')
    axs[0].set_xlabel("t [seconds]")
    axs[0].set_ylabel("$q_{c}$ [kg liquid/kg air]")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[0].grid(True)

    # Second subplot: q_r
    axs[1].plot(t, states[idx_test[j], :, 1], color='blue', marker='o', markersize=0.5, label=r'Exact, $q_r$')
    for k_alpha, alpha in enumerate(alphas):
        axs[1].fill_between(t[1:], lower[k_alpha, j, :, 1], upper[k_alpha, j, :, 1], 
                            color=colors[k_alpha], alpha=0.3, label=f"{100*(1-alpha)}% coverage")
    axs[1].plot(t[1:], rep[j, :, 1], color='red', marker='x', 
                markersize=0.5, label=r'SINDy (representative trajectory), $q_r$')
    axs[1].set_xlabel("t [seconds]")
    axs[1].set_ylabel("$q_{r}$ [kg liquid/kg air]")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].grid(True)

    fig.suptitle(f"Trajectory No. {idx_test[j]}, {method}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(file_path.removesuffix("_processed.npy")+"/cpSINDy/"+method+"/intervals"+str(idx_test[j])+".pdf")
    plt.close()