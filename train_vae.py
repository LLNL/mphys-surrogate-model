from src import data_utils as du, models, training
import xarray as xr
import torch
import numpy as np
import uuid

# parameters
batch_size = 500
n_latent = 3
epochs = 1000
init_lr = 1e-3
weight_decay = 1e-3
output_path = "./cnn_decoupled/"
id = uuid.uuid4().hex

# get data
#train_dataloader, test_dataloader, val_dataloader = du.create_dataloader("../pysdm_data/", batch_size)
ds_all = xr.open_dataset('data/box64.nc')
(data, norms, data_loaders) = du.create_e2e_dataloader(ds_all, cnn=True, batch_size=batch_size)
(train_dataloader, val_dataloader, test_dataloader) = data_loaders
(X, DX, T) = data


# set up the device
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

train_mse_recs = np.zeros(epochs)
val_mse_recs = np.zeros(epochs)

# define model
model = models.CNNAutoEncoder(n_channels=1, n_bins=63, n_latent=n_latent)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
	
	mod = training.train(model, train_dataloader, training.recon_loss, optimizer, device)
	train_mse = training.test(model, train_dataloader, training.recon_loss, device)
	val_mse = training.test(model, val_dataloader, training.recon_loss, device)

	train_mse_recs[epoch] = train_mse
	val_mse_recs[epoch] = val_mse

	if epoch%20 == 0:
		print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.8e}, Val. MSE: {val_mse:.8e}')

tmp_trainloss = "VAE_z{}_mse_{}_epochs_losses_{}.npz".format(n_latent, epochs, id)
np.savez(output_path + tmp_trainloss, train_loss=train_mse_recs, val_loss=val_mse_recs)

torch.save(model.state_dict(), output_path +  "model" + str(n_latent) + '_' + str(id) + ".pth")
print(f"Saved VAE with {n_latent} latent variables to model{n_latent}.pth")
