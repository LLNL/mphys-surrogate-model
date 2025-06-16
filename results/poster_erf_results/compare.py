# %%
import sys
import os
import numpy as np
import torch

project_root = os.path.relpath("../..")
sys.path.append(project_root)

from src import data_utils as du
from training_scripts.train_ae_ar import AEAutoregressor
from training_scripts.train_ae_sindy import AESINDy
from training_scripts.train_ae_NNdzdt import AENNdzdt
import matplotlib.pyplot as plt

# %%
(
    x_train,
    m_train,
    x_test,
    m_test,
    r_bins_edges,
    n_bins,
    dsd_time,
) = du.open_erf_dataset(
    sample_time=np.arange(0, 61, 5), path="../../data/congestus_coal_200m"
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
# %%
ae_sindy_pth = "./ae_sindy/model/FFNN_latent3_order3_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_0015405497354173a4eb3a26e57ab675.pth"
ae_sindy = AESINDy(n_bins=64, n_latent=3, poly_order=3)
ae_sindy.load_state_dict(torch.load(ae_sindy_pth, weights_only=True))
# %%
ae_nndzdt_pth = "./ae_nndzdt/model/FFNN_latent3_layers(40, 40, 40)_tr100_lr0.001_bs32_weights1.0-559.9560546875-55995.60546875_2103187c6f3943018c4ae375eef85cad.pth"
ae_nndzdt = AENNdzdt(
    n_channels=1, n_bins=64, n_latent=3, layer_size=(40, 40, 40), CNN=False
)
ae_nndzdt.load_state_dict(torch.load(ae_nndzdt_pth, weights_only=True))
# %%
ae_ar_pth = "ae_ar/model/erf_FFNN_latent3_order(10, 20, 10)_tr100_lr0.001_bs128_weights1-1_f74ecd55b87a43e2bfde2bf9973fdf10.pth"
ae_ar = AEAutoregressor(
    n_channels=1, n_bins=64, n_latent=3, n_lag=1, layer_size=(10, 20, 10), CNN=False
)
ae_ar.load_state_dict(torch.load(ae_ar_pth, weights_only=True))
# %%
test_ids = [0, 10, 20, 30]
tplt = [0, 5, 11]

(fig, ax) = plt.subplots(
    ncols=len(test_ids),
    nrows=len(tplt),
    figsize=(2.5 * len(test_ids), 2 * len(tplt)),
    sharey=True,
)

models = (ae_sindy, ae_nndzdt, ae_ar)
model_label = ["SINDy", "NN-driven", "AR"]
for k, model in enumerate(
    (
        ae_sindy,
        ae_nndzdt,
    )
):
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((3 + 1, 2))
    for il in range(3):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    z_encoded = model.encoder(torch.Tensor(x_test)).detach().numpy()
    for i, id in enumerate(test_ids):
        z0 = np.concatenate((z_encoded[id, 0, :], np.array([m_test[id, 0]])), axis=-1)
        latents_pred = du.simulate(z0, dsd_time[tplt], model.dzdt, zlim)
        x_pred = model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()

        for j, t in enumerate(tplt):
            if k == 0:
                ax[j][i].step(r_bins_edges, x_test[id, t, :], label="Data")
            ax[j][i].step(r_bins_edges, x_pred[j, :], label=model_label[k])

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Sample #{id}")
        ax[-1][i].set_xlabel("radius (um)")
n_lag = 1
for model in (ae_ar,):
    for i, id in enumerate(test_ids):
        x0 = x_test[id, :n_lag, :]
        m0 = m_test[id, 0]
        x_pred = np.zeros_like(x_test[id])
        x_pred[:n_lag, :] = (
            model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
        )
        for t in range(n_lag, x_test.shape[1]):
            x_pred[t, :] = (
                model(
                    torch.Tensor(x_pred[t - n_lag : t, :]).reshape(
                        -1, n_lag, x_pred[t].shape[0]
                    ),
                    torch.Tensor([m0]).reshape(1, 1, 1),
                )
                .detach()
                .numpy()[0][0]
            )

        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_pred[t, :], label="AR")


for j, t in enumerate(tplt):
    ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")

ax[0][0].legend()
plt.tight_layout()
plt.savefig("compare.pdf")
# %%
