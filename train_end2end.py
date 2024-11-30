import numpy as np
import data_utils as du
import xarray as xr
import torch
import pickle as pkl
import training

import tracemalloc

concat_files = False
#filepath = "box_data_64/*"
filepath = "box64.nc"

params = {}
# set up the device
device = torch.device(
	"cuda"
	if torch.cuda.is_available()
	# else "mps"
	# if torch.backends.mps.is_available()
	else "cpu"
    )
torch.backends.cudnn.benchmark = (
	True  # Used if inputs & model are constant, dropout?
    )
print(f"Using {device} device")

params["device"] = device
params["batch_size"] = 400
params['latent_dim'] = 3
params['poly_order'] = 1
params["library_size"] = params["poly_order"] * params["latent_dim"] + 1 # note this only holds for types of x, x^2; not xy
params['library_dim'] = 4
params["CNN"] = True
params["tracemalloc"] = True

params['loss_weight_recon'] = 1e2
params['loss_weight_sindy_z'] = 1.0
params['loss_weight_sindy_x'] = 1.0
params['loss_weight_sindy_reg'] = 1.0
params["pretraining_epochs"] = 100
params["training_epochs"] = 100
params["refinement_epochs"] = 20
params["learning_rate"] = 1e-3
params["patience"] = 50

if concat_files:
    ds_all = xr.open_mfdataset(filepath + ".nc", combine='nested', concat_dim='run')
    ds_all.load()
    ds_all = ds_all.fillna(0.0)
    ds_all = ds_all.rename_dims({'dv/dlnr_bin_index': 'mass_bin_idx'})
    ds_all = ds_all.rename_vars({'dv/dlnr': 'dvdlnr'})
    ds_all = ds_all.rename({'dv/dlnr_bin_index': 'mass_bin'})
    ds_all.to_netcdf('box64.nc')
else:
    ds_all = xr.open_dataset(filepath)

params['input_dim'] = len(ds_all['mass_bin'])
params["n_runs"] = len(ds_all['run'])
params["n_time"] = len(ds_all['time'])

(data, norms, data_loaders) = du.create_e2e_dataloader(ds_all, cnn=params["CNN"], batch_size=params["batch_size"])
(train_data, val_data, test_data) = data_loaders
(X, DX, T) = data

if params["tracemalloc"]:
    tracemalloc.start()

(vae, sindy_coeffs, loss, losses, val_loss, val_losses) = training.train_network_e2e(train_data, params, val_dataloader=val_data, device=params["device"])

if params["tracemalloc"]:
    # displaying the memory
    print("Memory Usage: Current (MB) | Peak (MB)")
    print([tracemalloc.get_traced_memory()[i] / 1024**2 for i in range(2)])

    # stopping the library
    tracemalloc.stop()


# SAVE
output_directory = "./end2end"
if params["CNN"]:
    case_name = "CNN_{}-{}-{}_lr{}_weights{}-{}-{}-{}".format(params["pretraining_epochs"], params["training_epochs"], params["refinement_epochs"], params["learning_rate"], params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"], params["loss_weight_sindy_reg"])
else:
    case_name = "FFNN_{}-{}-{}_lr{}_weights{}-{}-{}-{}".format(params["pretraining_epochs"], params["training_epochs"], params["refinement_epochs"], params["learning_rate"], params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"], params["loss_weight_sindy_reg"])

# total losses
with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump((loss, losses), pickle_file)
# sindy_coeffs
with open(output_directory + '/sindy/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump(sindy_coeffs, pickle_file)
# vae model
torch.save(vae.state_dict(), output_directory + '/autoencoder/' + case_name + ".pth")
print(f"Saved model and losses as {case_name}")
