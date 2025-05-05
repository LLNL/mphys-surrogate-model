import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
import pickle as pkl
import uuid
from src import data_utils as du, models, training, plotting
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(10)

num_epochs = 1
batch_size = 10
n_latent = 2
n_lag = 1  # default is 1
lr = 5e-4
wd = 1e-3
lr_sched = False
do_early_stopping = True
CNN = False
tol = 1e-8
w_recon = 1
w_dx = 1
w_dz = 1


class VAEAutoregressor(torch.nn.Module):
    def __init__(
        self,
        n_channels=2,
        n_bins=100,
        n_latent=10,
        n_lag=1,
        CNN=True,
    ):
        super(VAEAutoregressor, self).__init__()
        self.n_lag = n_lag
        if CNN:
            self.encoder = models.CNNEncoderVAE(
                n_channels=n_channels, n_bins=n_bins, n_latent=n_latent
            )
            self.decoder = models.CNNDecoder(
                n_channels=n_channels,
                n_bins=n_bins,
                n_latent=n_latent,
                distribution=True,
            )

        else:
            self.encoder = models.FFNNEncoderVAE(n_bins=n_bins, n_latent=n_latent)
            self.decoder = models.FFNNDecoder(
                n_bins=n_bins, n_latent=n_latent, distribution=True
            )
        self.autoregressor = models.Autoregressive(
            n_bins=n_latent + 1, n_bins_in=n_latent * n_lag + 1
        )

    def forward(self, bin0, M):
        latent0 = []
        for t in range(self.n_lag):
            latent0.append(self.encoder(bin0[:, t, :]).unsqueeze(1))
        latent0 = torch.cat(latent0, dim=2)
        # latent0 = self.encoder(bin0)
        latent0_M = torch.cat([latent0, M], dim=2)
        latent1_M = self.autoregressor(latent0_M)
        latent1 = latent1_M[:, :, :-1]
        bin1 = self.decoder(latent1)
        return bin1


# Open dataset
ds_all = xr.open_dataset("box64_train.nc")
dlnr = np.diff(np.log(ds_all["mass_bin"].values)).mean()
m_train = ds_all["dvdlnr"].sum(dim="mass_bin_idx")
x_train = (
    (ds_all["dvdlnr"] / m_train).transpose("run", "time", "mass_bin_idx").to_numpy()
)
m_scale = m_train.max()
m_train = (m_train / m_scale).to_numpy()
n_bins = x_train.shape[2]

ds_test = xr.open_dataset("box64_test.nc")
m_test = ds_test["dvdlnr"].sum(dim="mass_bin_idx")
x_test = (
    (ds_test["dvdlnr"] / m_test).transpose("run", "time", "mass_bin_idx").to_numpy()
)
m_test = (m_test / m_scale).to_numpy()


# Create torch dataset
class NormedBinDataset1C(Dataset):
    def __init__(self, dmdlnr_normed, M, lag=1):
        self.nbin = dmdlnr_normed.shape[2]
        self.lag = lag
        self.bin0 = (
            []
        )  # dmdlnr_normed.astype(np.float32)[:,:-1*lag,:].reshape([-1, 1, self.nbin])
        self.bin1 = (
            []
        )  # dmdlnr_normed.astype(np.float32)[:,lag:,:].reshape([-1, 1, self.nbin])
        self.M = []  # M.astype(np.float32).reshape([-1, 1, 1])

        for i in range(dmdlnr_normed.shape[1] - lag):
            self.bin0.append(dmdlnr_normed[:, i : i + lag, :].astype(np.float32))
            self.bin1.append(dmdlnr_normed[:, i + lag, :].astype(np.float32))
            self.M.append(M[:, i + lag].astype(np.float32))

        self.bin0 = np.array(self.bin0).reshape([-1, lag, self.nbin])
        self.bin1 = np.array(self.bin1).reshape([-1, 1, self.nbin])
        self.M = np.array(self.M).reshape([-1, 1, 1])

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :], self.bin1[idx, :], self.M[idx]


# Initialize the model
model = VAEAutoregressor(
    n_channels=1, n_bins=n_bins, n_latent=n_latent, n_lag=n_lag, CNN=CNN
)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
early_stopping = training.EarlyStopping(patience=20)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Training loop
# Convert data to batches
# train_data = torch.utils.data.TensorDataset(inputs, outputs)
train_data = NormedBinDataset1C(x_train, m_train, lag=n_lag)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
# test_data = torch.utils.data.TensorDataset(test_inputs, test_outputs)
test_data = NormedBinDataset1C(x_test, m_test, lag=n_lag)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=x_test.shape[0], shuffle=True
)

losses = []
recon_losses = []
dx_losses = []
dz_losses = []

test_losses = []
test_recon_losses = []
test_dx_losses = []
test_dz_losses = []


for epoch in range(num_epochs):
    # train
    model.train()
    for batch_X, batch_y, batch_M in train_loader:
        # Forward pass
        pred_y = model(batch_X, batch_M)
        pred_z = model.encoder(batch_X)
        pred_z1 = model.autoregressor(
            torch.cat((pred_z.reshape(-1, 1, n_latent * n_lag), batch_M), dim=2)
        )  # note: can train AR to predict zero change in M, or just ignore M in training
        data_z1 = torch.cat((model.encoder(batch_y), batch_M), dim=2)
        pred_x_recon = model.decoder(model.encoder(batch_X))

        loss_dx = divergence(torch.log(pred_y + tol), torch.log(batch_y + tol))
        loss_dz = criterion(pred_z1, data_z1)
        loss_recon = divergence(torch.log(pred_x_recon + tol), torch.log(batch_X + tol))
        loss = w_dx * loss_dx + w_recon * loss_recon + w_dz * loss_dz

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    recon_losses.append(loss_recon.item())
    dx_losses.append(loss_dx.item())
    dz_losses.append(loss_dz.item())

    # test
    model.eval()
    for batch_X, batch_y, batch_M in test_loader:
        # Forward pass
        pred_y = model(batch_X, batch_M)
        pred_z = model.encoder(batch_X)
        pred_z1 = model.autoregressor(
            torch.cat((pred_z.reshape(-1, 1, n_latent * n_lag), batch_M), dim=2)
        )  # note: can train AR to predict zero change in M, or just ignore M in training
        data_z1 = torch.cat((model.encoder(batch_y), batch_M), dim=2)
        pred_x_recon = model.decoder(model.encoder(batch_X))

        loss_dx = divergence(torch.log(pred_y + tol), torch.log(batch_y + tol))
        loss_dz = criterion(pred_z1, data_z1)
        loss_recon = divergence(torch.log(pred_x_recon + tol), torch.log(batch_X + tol))
        loss = w_dx * loss_dx + w_recon * loss_recon + w_dz * loss_dz

    test_losses.append(loss.item())
    test_recon_losses.append(loss_recon.item())
    test_dx_losses.append(loss_dx.item())
    test_dz_losses.append(loss_dz.item())

    if epoch % 1 == 0:
        print(
            f"Epoch [{epoch}/{num_epochs}], Train Loss: {losses[-1]:.4f} |  Test Loss: {test_losses[-1]:.4f} | LR: {sched.get_last_lr()}"
        )

    if lr_sched:
        sched.step(loss / len(test_loader))
    if do_early_stopping:
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

# Export/save
output_directory = "trained_models/vae_autoregressor_normed"
if CNN:
    case_name = f"CNN_AdamW_L2_lr{lr}_bs{batch_size}_ne{num_epochs}_" + uuid.uuid4().hex
else:
    case_name = f"FFNN_lr{lr}_bs{batch_size}_ne{num_epochs}_" + uuid.uuid4().hex

with open(output_directory + "/losses/" + case_name + ".pkl", "wb") as pickle_file:
    pkl.dump(
        (
            losses,
            recon_losses,
            dx_losses,
            dz_losses,
            test_losses,
            test_recon_losses,
            test_dx_losses,
            test_dz_losses,
        ),
        pickle_file,
    )

# vae model
torch.save(model.state_dict(), output_directory + "/model/" + case_name + ".pth")
print(f"Saved model and losses as {case_name}")

############### PLOTS PLOTS PLOTS ###############
# Plot training loss
plotting.plot_losses(
    losses,
    test_losses=test_losses,
    sub_losses=[dx_losses, dz_losses, recon_losses],
    labels=["X: t -> t+1", "Z: t -> t+1", "Recon"],
    title=f"Training Loss, lag {n_lag}",
    saveas=output_directory + "/plots/" + case_name + "_losses.png",
)

# Plot distributions: reconstruction
r_bins_edges = ds_all["mass_bin"]
test_ids = [0, 20, 30, 40]
plotting.plot_reconstructions(
    model,
    test_ids,
    x_test,
    r_bins_edges,
    saveas=output_directory + "/plots/" + case_name + "_reconstructions.png",
)

# Predictions: Multi time step
tplt = [3, 5, 8, 12]  # [0, 1, 2, 3, 5, 8, 12]
plotting.plot_predictions_AE_AR(
    model,
    test_ids,
    tplt,
    x_test,
    m_test,
    r_bins_edges,
    saveas=output_directory + "/plots/" + case_name + "_predictions.png",
)

# Plot trajectories of the latent variables
plotting.plot_latent_trajectories_AR(
    n_latent,
    model,
    x_test,
    m_test,
    saveas=output_directory + "/plots/" + case_name + "_trajectories.png",
)
plotting.viz_3d_latent_space(
    model,
    x_test,
    ds_test["time"].to_numpy() / 1e9,
    "trained_models/vae_autoregressor_normed/plots",
    case_name,
)
