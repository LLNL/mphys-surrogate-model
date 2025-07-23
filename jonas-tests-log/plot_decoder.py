import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.append('/g/g14/katona1/mphys-surrogate-model')

from src import data_utils as du

path = '/g/g14/katona1/mphys-surrogate-model/jonas-tests-log/cp/decoder'

(
    x_train,
    m_train,
    x_test,
    m_test,
    r_bins_edges,
    n_bins,
    dsd_time,
) = du.open_erf_dataset(
    sample_time=np.arange(0, 61, 5), path="/g/g14/katona1/mphys-surrogate-model/data/congestus_coal_200m"
)

# load results from vanilla conformal predictions
with open(path+'/vanilla.pkl', 'rb') as f:
    alphas, DSD_bands_full = pickle.load(f)

# load results from split conformal predictions
with open(path+'/split.pkl', 'rb') as f:
    _, idx_testing, DSD_bands_split  = pickle.load(f)
    
test_ids_raw = [0, 10, 20, 30] # indices for trajectories to plot, indexed according to split conformal
test_ids = idx_testing[test_ids_raw] # trajectories to plot
tplt = [0, 5, 11] # which times to plot

(fig, ax) = plt.subplots(
    ncols=len(test_ids),
    nrows=len(tplt),
    figsize=(2.5 * len(test_ids), 2 * len(tplt)),
    sharey=True,
)

model_label = ["SINDy", "NN-driven", "AR"] # model labels

lower = DSD_bands_full[0]
rep = DSD_bands_full[1]
upper = DSD_bands_full[2]

# use alphas to get color arguments for fill_between (the prediction intervals on the plot) 
cmap    = plt.get_cmap('autumn_r')     # 0→yellow, 1→red
colors  = cmap(np.tanh(np.pi*np.array(alphas)))            # take tanh to ensure that it is mostly red until very close to 0

for k, model in enumerate(model_label):
    for i, id in enumerate(test_ids):
        for j, t in enumerate(tplt):
            if k == 0:
                ax[j][i].step(r_bins_edges, x_test[id, t], label="Data")
            for k_alpha, alpha in enumerate(alphas):
                if k == 0:
                    ax[j][i].fill_between(r_bins_edges, lower[k, k_alpha, id, t], upper[k, k_alpha, id, t], step='pre',
                        color=colors[k_alpha], alpha=0.3, label=f"{100*(1-alpha)}% coverage") 
                else:
                    ax[j][i].fill_between(r_bins_edges, lower[k, k_alpha, id, t], upper[k, k_alpha, id, t], step='pre',
                        color=colors[k_alpha], alpha=0.3)
            ax[j][i].step(r_bins_edges, rep[k, id, t], label=model) # representative band
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Sample #{id}")
        ax[-1][i].set_xlabel("radius (um)")
for j, t in enumerate(tplt):
    ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
ax[0][0].legend()
fig.suptitle("Logarithmic full conformal predictions", fontsize=14)
plt.tight_layout()
plt.savefig("/g/g14/katona1/mphys-surrogate-model/jonas-tests-log/compare.pdf")