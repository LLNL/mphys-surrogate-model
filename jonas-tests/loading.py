import os 
import sys
import torch
import numpy as np

sys.path.append('/g/g14/katona1/mphys-surrogate-model')

from src import data_utils as du
from training_scripts.train_ae_ar import AEAutoregressor
from training_scripts.train_ae_sindy import AESINDy
from training_scripts.train_ae_NNdzdt import AENNdzdt
import matplotlib.pyplot as plt

# %%
(
    x_train, # (#samples, #timesteps, #dims)
    m_train, # (#samples, #timesteps)
    x_test,
    m_test,
    r_bins_edges,
    n_bins,
    dsd_time,
) = du.open_erf_dataset(
    sample_time=np.arange(0, 61, 5), path="/g/g14/katona1/mphys-surrogate-model/data/congestus_coal_200m"
)

params = {
    "data_src": "erf",
    "random_seed": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "poly_order": 3,
    "n_lag": 1,
    "lr_sched": True,
    "patience": 50,
    "tol": 1e-8,
    "wd": 1e-3,
    "lambda1_factor": 0.5,
    # "lambda3_sparsity": 0.0, TODO: sequential thresholding
    "CNN": False,
    "print_frequency": 1,
}

main_path = '/g/g14/katona1/mphys-surrogate-model/results/poster_erf_results/'
ae_ar_checkpoint = torch.load(main_path + 'ae_ar/model/erf_FFNN_latent3_order(10, 20, 10)_tr100_lr0.001_bs128_weights1-1_f74ecd55b87a43e2bfde2bf9973fdf10.pth',
                              weights_only=True)
ae_nndzdt_checkpoint = torch.load(main_path + 'ae_nndzdt/model/FFNN_latent3_layers(40, 40, 40)_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_7850dd2260f34875baf55452f4311d60.pth',
                              weights_only=True)
ae_sindy_checkpoint = torch.load(main_path + 'ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth',
                              weights_only=True)

# %%
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(ae_sindy_checkpoint)
# %%
ae_nndzdt = AENNdzdt(
    n_channels=1, n_bins=64, n_latent=3, layer_size=(40, 40, 40), CNN=False
)
ae_nndzdt.load_state_dict(ae_nndzdt_checkpoint)
# %%
ae_ar = AEAutoregressor(
    n_channels=1, n_bins=64, n_latent=3, n_lag=1, layer_size=(10, 20, 10), CNN=False
)
ae_ar.load_state_dict(ae_ar_checkpoint)

models = (ae_sindy, ae_nndzdt, ae_ar)
model_label = ["SINDy", "NN-driven", "AR"]

print(x_train.shape)
print(m_train.shape)
exit()

def run_ae_X(x, m):
    # DSD data, decoded (predictions from network)
    DSD_all = np.empty((len(model_label),)+x.shape, dtype=float) 
    # mass data 
    M_all = np.empty((len(model_label),)+m.shape, dtype=float) 
    for k, model in enumerate(
        (
            ae_sindy,
            ae_nndzdt,
        )
    ):
        z_enc_train = model.encoder(torch.Tensor(x)).detach().numpy()
        zlim = np.zeros((3 + 1, 2)) # DSD bins
        for il in range(3):
            zlim[il][0] = z_enc_train[:, :, il].min()
            zlim[il][1] = z_enc_train[:, :, il].max()
        zlim[-1][0] = m.min()
        zlim[-1][1] = m.max()

        for id in range(len(x)): # loop through each initial condition/sample/gridbox
            z0 = np.concatenate((z_enc_train[id, 0, :], np.array([m[id, 0]])), axis=-1)
            latents_pred = du.simulate(z0, dsd_time, model.dzdt, zlim)
            DSD_all[k, id] = model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy() # add DSD predictions
            M_all[k, id] = latents_pred[:, -1] # add mass predictions
    n_lag = 1
    for model in (ae_ar,):
        for id in range(len(x)): # loop through each initial condition/sample/gridbox
            x0 = x[id, :n_lag, :]
            m0 = m[id, 0]
            DSD_all[2, id][:n_lag, :] = (
                model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
            )
            for t in range(n_lag, x.shape[1]):
                DSD_all[2, id][t, :] = (
                    model(
                        torch.Tensor(DSD_all[2, id][t - n_lag : t, :]).reshape(
                            -1, n_lag, DSD_all[2, id][t].shape[0]
                        ),
                        torch.Tensor([m0]).reshape(1, 1, 1),
                    )
                    .detach()
                    .numpy()[0][0]
                )
    return DSD_all

def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    """
    Compute all lower- and upper-tail quantiles in one shot.

    residuals : array_like, shape (N, T, D)
    alpha_lows: list of alpha/2 levels (e.g. [0.125, 0.025] for 75% & 95%)
    alpha_ups : same as alpha_lows
    Returns
    -------
    q_low, q_high : arrays of shape (len(alpha_lows), T, D)
    """
    import numpy as _np

    # 1) build the full list of levels
    lows  = _np.array(alpha_lows)
    ups   = 1.0 - _np.array(alpha_ups)
    all_q = _np.concatenate([lows, ups])       # e.g. [0.125, 0.025, 0.875, 0.975]

    # 2) sort levels and remember how to invert
    sort_idx = _np.argsort(all_q)
    q_sorted = all_q[sort_idx]

    # 3) single quantile call
    qs = _np.quantile(residuals, q_sorted, axis=0)

    # 4) invert the sort
    qs_unsorted = _np.empty_like(qs)
    qs_unsorted[sort_idx] = qs

    # 5) split into lows / highs
    m = len(alpha_lows)
    q_low  = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high

alphas = [0.25, 0.1, 0.05, 0.01]

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a/2 for a in alphas]
alpha_ups  = alpha_lows.copy()

# full
# loop through all models to generate
lower_full = np.empty((len(models),len(alphas),)+x_test.shape, dtype=float)
upper_full = lower_full.copy()
print('Predicting the training data.')
DSD_train_all = run_ae_X(x_train, m_train)
print('Predicting the testing data.')
pred_test = run_ae_X(x_test, m_test)
print('Running vanilla conformal predictions.')
for i, m in enumerate(models): 
    res_signed = DSD_train_all[i] - x_train
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups) 
    lower_full[i] = pred_test[i, np.newaxis, ...] - q_high[:, np.newaxis, ...]
    upper_full[i] = pred_test[i, np.newaxis, ...] - q_low [:, np.newaxis, ...]
    
# split
print('Splitting testing data into calibration and testing data. 50-50 split.')
# make a global index array
n_all  = len(x_test)
idx_all = np.arange(n_all)            # [0,1,...,n_all-1]

from sklearn.model_selection import train_test_split

# calibration/test split on indices
idx_cal, idx_test = train_test_split(
    idx_all,
    test_size=0.5,
    random_state=1952
)

# split data into training and testing data (training includes calibration, FYI)
x_testing = x_test.copy()
m_testing = m_test.copy()
x_cal  = x_testing[idx_cal]
x_testing   = x_testing[idx_test]
m_cal = m_testing[idx_cal]
m_testing  = m_testing[idx_test]
lower_split = np.empty((len(models),len(alphas),)+x_testing.shape, dtype=float)
upper_split = lower_split.copy()
print('Predicting the calibration data.')
pred_cal= run_ae_X(x_cal, m_cal)
print('Predicting the testing data.')
pred_testing = run_ae_X(x_testing, m_testing)
print('Running split conformal predictions.')
for i, m in enumerate(models): 
    res_signed = pred_cal[i] - x_cal
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups) 
    lower_split[i] = pred_testing[i, np.newaxis, ...] - q_high[:, np.newaxis, ...]
    upper_split[i] = pred_testing[i, np.newaxis, ...] - q_low [:, np.newaxis, ...]
    
