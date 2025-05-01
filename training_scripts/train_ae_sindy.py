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

torch.manual_seed(0)
tol=1e-8

params = {}

device = torch.device(
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
    )
torch.backends.cudnn.benchmark = (
	True  # Used if inputs & model are constant, dropout?
    )
print(f"Using {device} device")

params["device"] = device
params["training_epochs"] = 100
params["batch_size"] = 100
params["learning_rate"] = 1e-3
params['latent_dim'] = 3
params['poly_order'] = 2 # = "BB" for FFNN dzdt version
params['loss_weight_recon'] = 1e0
params['loss_weight_sindy_z'] = 1e2
params['loss_weight_sindy_x'] = 1e6
params["CNN"] = False
params["patience"] = 50
#params["layers"] = (10, 20, 10) # only used for the Black-Box dzdt
print(params)

class AESINDy(torch.nn.Module):
    def __init__(self,n_channels=1,n_bins=100,n_latent=10,poly_order=2,CNN=True):
        super(AESINDy, self).__init__()

        if CNN:
            self.encoder = models.CNNEncoderVAE(n_channels=n_channels, n_bins=n_bins, n_latent=n_latent)
            self.decoder = models.CNNDecoder(n_channels=n_channels, n_bins=n_bins, n_latent=n_latent, distribution=True)

        else:
            self.encoder = models.FFNNEncoderVAE(n_bins=n_bins, n_latent=n_latent)
            self.decoder = models.FFNNDecoder(n_bins=n_bins, n_latent=n_latent, distribution=True)

        self.dzdt = models.SINDyDeriv(n_latent=n_latent + 1, poly_order=poly_order)

    def forward(self, bin0, M):
        z0 = self.encoder(bin0)
        dzMdt = self.dzdt(z0, M)
        dzdt = dzMdt[:, :, :-1]
        return dzdt

# Open dataset
ds_all = xr.open_dataset('../data/box64_train.nc')
dlnr = np.diff(np.log(ds_all['mass_bin'].values)).mean()
m_train = ds_all['dvdlnr'].sum(dim='mass_bin_idx')
x_train = (ds_all['dvdlnr'] / m_train).transpose('run','time','mass_bin_idx').to_numpy()
m_scale = m_train.max()
m_train = (m_train / m_scale).to_numpy()
n_bins = x_train.shape[2]
time = (ds_all['time'] / np.timedelta64(1, 's')).to_numpy()

ds_test = xr.open_dataset('../data/box64_test.nc')
m_test = ds_test['dvdlnr'].sum(dim='mass_bin_idx')
x_test = (ds_test['dvdlnr'] / m_test).transpose('run','time','mass_bin_idx').to_numpy()
m_test = (m_test / m_scale).to_numpy()

# Create torch dataset
class NormedBinDatasetSINDy(Dataset):
    def __init__(self, dmdlnr_normed, time, M):
        self.nbin = dmdlnr_normed.shape[2]
        self.t = time
        self.dt = self.t[1] - self.t[0]
        self.x = dmdlnr_normed.reshape(-1, 1, self.nbin)
        self.dx = np.gradient(dmdlnr_normed, axis=1).reshape(-1, 1, self.nbin) / self.dt
        self.M   = M.reshape(-1, 1, 1)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx, :], self.dx[idx,:], self.M[idx]

train_data = NormedBinDatasetSINDy(x_train, time, m_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
test_data = NormedBinDatasetSINDy(x_test, time, m_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=x_test.shape[0], shuffle=True)

# Initialize the model
model = AESINDy(n_channels=1, n_bins=n_bins, n_latent=params["latent_dim"], poly_order=params["poly_order"], CNN=params["CNN"])

# Loss function and optimizer
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-3)
early_stopping = training.EarlyStopping(patience=params["patience"])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

losses = []
recon_losses = []
dx_losses = []
dz_losses = []

test_losses = []
test_recon_losses = []
test_dx_losses = []
test_dz_losses = []

for epoch in range(params["training_epochs"]):
    # train
    model.train()
    for batch_x, batch_dx, batch_M in train_loader:
        # Forward pass
        pred_x_recon = model.decoder(model.encoder(batch_x))
        z = model.encoder(batch_x)
        zz = z.clone().detach().requires_grad_()
        if params["CNN"]:
            grad_enc_x = torch.func.vmap(torch.func.jacrev(model.encoder, chunk_size=20),
                                        chunk_size=20)(batch_x)[:, 0, :, :, 0, :]
            grad_dec_z = torch.func.vmap(torch.func.jacrev(model.decoder, chunk_size=20),
                                        chunk_size=20)(zz)[:, 0, :, :, 0, :]
        else:
            grad_enc_x = torch.func.vmap(
                torch.func.jacrev(model.encoder, chunk_size=20),
                chunk_size=20)(batch_x)[:, :, :, 0, :]
            grad_dec_z = torch.func.vmap(
                torch.func.jacrev(model.decoder, chunk_size=20),
                chunk_size=20)(zz)[:, :, :, 0, :]

        dz = torch.einsum('abcd, abd->abc', grad_enc_x, batch_dx)
        pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
        pred_dx = torch.einsum('abcd, abd->abc', grad_dec_z, pred_dz)

        loss_recon = divergence(torch.log(pred_x_recon + tol), torch.log(batch_x + tol))
        loss_dx = criterion(pred_dx, batch_dx)
        loss_dz = criterion(pred_dz, dz)

        loss = params["loss_weight_sindy_x"] * loss_dx + params["loss_weight_recon"] * loss_recon + params["loss_weight_sindy_z"] * loss_dz

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    losses.append(loss.item())
    recon_losses.append(loss_recon.item())
    dx_losses.append(loss_dx.item())
    dz_losses.append(loss_dz.item())

    # test
    model.eval()
    for batch_x, batch_dx, batch_M in test_loader:
        # Forward pass
        pred_x_recon = model.decoder(model.encoder(batch_x))
        z = model.encoder(batch_x)
        zz = z.clone().detach().requires_grad_()
        if params["CNN"]:
            grad_enc_x = torch.func.vmap(torch.func.jacrev(model.encoder, chunk_size=20),
                                         chunk_size=20)(batch_x)[:, 0, :, :, 0, :]
            grad_dec_z = torch.func.vmap(torch.func.jacrev(model.decoder, chunk_size=20),
                                         chunk_size=20)(zz)[:, 0, :, :, 0, :]
        else:
            grad_enc_x = torch.func.vmap(
                torch.func.jacrev(model.encoder, chunk_size=20),
                chunk_size=20)(batch_x)[:, :, :, 0, :]
            grad_dec_z = torch.func.vmap(
                torch.func.jacrev(model.decoder, chunk_size=20),
                chunk_size=20)(zz)[:, :, :, 0, :]

        dz = torch.einsum('abcd, abd->abc', grad_enc_x, batch_dx)
        pred_dz = model.dzdt(z, batch_M)[:, :, :-1]
        pred_dx = torch.einsum('abcd, abd->abc', grad_dec_z, pred_dz)

        loss_recon = divergence(torch.log(pred_x_recon + tol), torch.log(batch_x + tol))
        loss_dx = criterion(pred_dx, batch_dx)
        loss_dz = criterion(pred_dz, dz)

        loss = params["loss_weight_sindy_x"] * loss_dx + params["loss_weight_recon"] * loss_recon + params[
            "loss_weight_sindy_z"] * loss_dz

    test_losses.append(loss.item())
    test_recon_losses.append(loss_recon.item())
    test_dx_losses.append(loss_dx.item())
    test_dz_losses.append(loss_dz.item())

    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{params['training_epochs']}], Train Loss: {losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}") # | LR: {sched.get_last_lr()}")
        print(f"Recon: {params['loss_weight_recon'] * recon_losses[-1]:.4f} | "
              f"dx: {params['loss_weight_sindy_x'] * dx_losses[-1]:.4f} | "
              f"dz: {params['loss_weight_sindy_z'] *dz_losses[-1]:.4f} |")
    early_stopping(loss)
    if early_stopping.early_stop:
        print("Training stopped early.")
        break

# SAVE
output_directory = "../trained_models/ae_sindy_normed"
prefix = "CNN"
case_name = prefix + "_latent{}_order{}_tr{}_lr{}_bs{}_weights{}-{}-{}_{}".format(
        params["latent_dim"], params["poly_order"],
        params["training_epochs"], params["learning_rate"], params["batch_size"],
        params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"],
        uuid.uuid4().hex)

# total losses
with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump((loss, losses), pickle_file)
torch.save(model.state_dict(), output_directory + '/model/' + case_name + ".pth")
print(f"Saved model and losses as {case_name}")


############### PLOTS PLOTS PLOTS ###############
# Plot training loss
plotting.plot_losses(losses, test_losses=test_losses,
                     sub_losses=[params["loss_weight_sindy_x"] * np.array(dx_losses),
                                 params["loss_weight_sindy_z"] * np.array(dz_losses),
                                 params["loss_weight_recon"] * np.array(recon_losses)],
                     labels=["dx/dt", "dz/dt", "Recon"], title=f"Training Loss")

# Plot distributions: reconstruction
r_bins_edges = ds_all['mass_bin']
test_ids = [0, 20, 30, 40]
plotting.plot_reconstructions(model, test_ids, x_test, r_bins_edges)

# Predictions: Multi time step
tplt = [3, 5, 8, 12] #[0, 1, 2, 3, 5, 8, 12]
plotting.plot_predictions_AE_SINDy(test_ids, tplt,
                                   params["latent_dim"], params["poly_order"],
                                   model, x_test, m_test,
                                    x_train, m_train, r_bins_edges)


# Plot trajectories of the latent variables
plotting.plot_latent_trajectories_SINDy(params["latent_dim"], params["poly_order"],
                                        model, x_test, m_test,
                                        test_data.dt, test_data.t,
                                        x_train, m_train,
                                        plt_dx = True
                                        )