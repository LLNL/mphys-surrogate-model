import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from lasdi.gp import eval_gp, fit_gps
from lasdi.gplasdi import average_rom, sample_roms
from lasdi.latent_dynamics.sindy import SINDy
from lasdi.latent_dynamics.wsindy import \
    wSINDy  # only if you are on wSINDy branch
from lasdi.latent_space import Autoencoder
from lasdi.physics import Physics
from src import models, plotting

ae_weight = 1e0
sindy_weight = 1e2
coef_weight = 0.0  # 1e-6
lr = 1e-3
n_iter = 1000
hidden_units = [32, 16, 8]
n_z = 3  # latent dim
batch_size = 100

torch.manual_seed(9263314059755493460)
# 6170738560895651907
print(torch.seed())
np.random.rand(0)

# Open dataset
ds_train = xr.open_dataset("../data/box64_train.nc")
ds_test = xr.open_dataset("../data/box64_test.nc")
X_train = torch.Tensor(
    (ds_train["dvdlnr"])
    .transpose("run", "time", "mass_bin_idx")  # .isel(run=slice(None,100))
    .to_numpy()
)
X_test = torch.Tensor(
    (ds_test["dvdlnr"]).transpose("run", "time", "mass_bin_idx").to_numpy()
)
x_norm = X_train.max()
X_train /= x_norm
X_test /= x_norm
n_sims = X_train.shape[0]
n_time = X_train.shape[1]
n_bins = X_train.shape[2]
r_bins = ds_train["mass_bin"].to_numpy()
t = (ds_train["time"] / np.timedelta64(1, "s")).to_numpy()
sol_dim = 1

# Placeholder
param_train = torch.Tensor([[0.01 * (np.random.rand() - 0.5)] for _ in range(n_sims)])
param_test = torch.Tensor(
    [[0.01 * (np.random.rand() - 0.5)] for _ in range(X_test.shape[0])]
)


class CustomPhysicsModel(Physics):
    def __init__(self):
        self.dim = 1
        self.nt = n_time
        self.dt = t[1] - t[0]
        self.qdim = sol_dim
        self.qgrid_size = [n_bins]
        self.t_grid = t
        return

    """ See lasdi.physics.Physics class for necessary subroutines """


physics = CustomPhysicsModel()

# Autoencoder Definition; You could also define your own custom autoencoder
ae_cfg = {"hidden_units": hidden_units, "latent_dimension": n_z, "activation": "ReLU"}
# autoencoder = Autoencoder(physics, ae_cfg)
autoencoder = models.FFNNAutoEncoder(n_bins=n_bins, n_latent=n_z)

# Initialize latent dynamics
# sindy_options = {
#     "sindy": {"fd_type": "sbp12", "coef_norm_order": 2}
# }  # finite-difference operator for computing time derivative of latent trajectory.
# ld = SINDy(autoencoder.n_z, physics.nt, sindy_options)
# For WSINDY:
wsindy_options = {
    "wsindy": {
        "fd_type": "sbp12",
        "coef_norm_order": 2,
        "LS_loss_type": "weak",
        "pq": 8,
    }
}
ld = wSINDy(n_z, physics.nt, physics.dt, wsindy_options)


# Training
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

best_loss = np.inf
d_ae = autoencoder.to(device)
d_Xtrain = X_train.to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
MSE = torch.nn.MSELoss()

loss_hist = np.zeros([n_iter, 4])
grad_hist = np.zeros([n_iter, 4])
tic_start = time.time()

for epoch in range(n_iter):
    batch_ids = np.random.choice(X_train.shape[0], batch_size)
    optimizer.zero_grad()
    d_Z = d_ae.encoder(d_Xtrain[batch_ids])
    d_Xpred = d_ae.decoder(d_Z)
    Z = d_Z.cpu()

    loss_ae = MSE(d_Xpred, d_Xtrain[batch_ids])
    coefs, loss_sindy, loss_coef = ld.calibrate_all(
        Z, physics.dt, compute_loss=True, numpy=False
    )
    max_coef = torch.max(torch.abs(coefs))

    loss = (
        ae_weight * loss_ae
        + sindy_weight * loss_sindy / n_sims
        + coef_weight * loss_coef / n_sims
    )
    loss_hist[epoch] = [
        loss.item(),
        loss_ae.item(),
        loss_sindy.item(),
        loss_coef.item(),
    ]

    loss.backward()

    optimizer.step()

    if (
        loss.item() < best_loss
    ):  # TODO(emily): port over some of the checkpoint/saving stuff
        d_ae = autoencoder.to(device)
        best_loss = loss.item()
        best_coefs = coefs

    print(
        "Iter: %05d/%d, Loss: %.5e, Loss AE: %.5e, Loss SI: %.5e, Loss COEF: %.5e, max|c|: %.5e"
        % (
            epoch + 1,
            n_iter,
            loss.item(),
            loss_ae.item(),
            loss_sindy.item(),
            loss_coef.item(),
            max_coef,
        )
    )

coefs = ld.calibrate_all(Z, physics.dt, compute_loss=False, numpy=True)
print(coefs)

# PLOTTING
autoencoder = d_ae.cpu()
Z = autoencoder.encoder(X_train)
Z_test = autoencoder.encoder(X_test)
X_AE_recon = autoencoder(torch.Tensor(X_test)).detach().numpy()

# plot losses
plotting.plot_losses(
    loss_hist[:, 0],
    sub_losses=[
        loss_hist[:, 1],
        loss_hist[:, 2],
        loss_hist[:, 3],
    ],
    labels=["recon", "dz/dt", "reg"],
    title=f"Training Loss",
    # saveas=output_directory + "/plots/" + case_name + "_losses.png",
)

# Plot distributions: reconstruction
test_ids = [0, 10, 20, 25]
plotting.plot_reconstructions(
    d_ae,
    test_ids,
    X_test,
    r_bins,
)

# Predictions: Multi time step
tplt = [3, 5, 8, 12]  # [0, 1, 2, 3, 5, 8, 12]
(fig, ax) = plt.subplots(
    ncols=len(test_ids), nrows=len(tplt), figsize=(3 * len(test_ids), 6)
)
for i, id in enumerate(test_ids):
    Z0 = d_ae.encoder(torch.Tensor(X_test[id, 0])).detach().numpy()
    Zi = ld.simulate(coefs, Z0, tplt)
    x_plt = d_ae.decoder(torch.Tensor(Zi)).detach().numpy()
    for j, ti in enumerate(tplt):
        ax[j][i].step(r_bins, X_test[id, ti])
        ax[j][i].step(r_bins, x_plt[j])
        ax[j][i].set_xscale("log")
plt.show()

# Predictions: latent space
(fig, ax) = plt.subplots(ncols=n_z, figsize=(3 * n_z, 3))
colors = ["blue", "orange", "green", "pink", "purple", "gray"]
for i, id in enumerate(test_ids):
    Z0 = d_ae.encoder(torch.Tensor(X_test[id, 0])).detach().numpy()
    Zi = ld.simulate(coefs, Z0, t)
    Zt = d_ae.encoder(torch.Tensor(X_test[id, :])).detach().numpy()
    for j in range(n_z):
        ax[j].plot(t, Zi[:, j], ls="--", label="Model", color=colors[i])
        ax[j].plot(t, Zt[:, j], label="True", color=colors[i])
plt.show()
