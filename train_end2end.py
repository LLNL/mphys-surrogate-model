import numpy as np
import pickle as pkl
import random
import xarray as xr
import torch
import training

# Load the data
filepath = "box_data/*"
ds_all = xr.open_mfdataset(filepath + ".nc", combine='nested', concat_dim='run')
ds_all.load()
one_sec = np.timedelta64(1, 's')
ds_all = ds_all.fillna(0.0)

n_ics = ds_all.run.size
n_steps = ds_all.time.size
t = (ds_all['time'] / one_sec).to_numpy()
dt = int((ds_all['time'].isel(time=1) - ds_all['time'].isel(time=0)) / one_sec)
x = ds_all['dv/dlnr'].transpose('run','time','dv/dlnr_bin_index').to_numpy()

# shuffle data
shuffle_idx = ds_all['run'].data
random.shuffle(shuffle_idx)
x = x[shuffle_idx, :, :]

normalization = np.max(x)
# normalization
if normalization is not None:
    x = x / normalization
dx = np.gradient(x, axis=1) / dt

data = {}
data['t'] = t
data['x'] = torch.tensor(x, dtype=torch.float32, requires_grad=True)
data['dx'] = torch.tensor(dx)


params = {}
params['input_dim'] = data['x'].shape[2]
params['latent_dim'] = 3
params['poly_order'] = 1
params["library_size"] = params["poly_order"] * params["latent_dim"] + 1 # note this only holds for types of x, x^2; not xy
params['library_dim'] = 4
params["n_runs"] = data['x'].shape[0]
params["n_time"] = data['x'].shape[1]
data['z'] = np.random.rand(params["n_runs"], params["n_time"], params["latent_dim"])

params['loss_weight_recon'] = 1.0
params['loss_weight_sindy_z'] = 10.0
params['loss_weight_sindy_x'] = 1.0
params['loss_weight_sindy_reg'] = 1.0
# consider normalizing the total loss weights to sum to 1
params["max_epochs"] = 500
params["learning_rate"] = 1e-5
# consider batching and/or KFold cross validation


# TRAINING
(vae, sindy_coeffs, loss, losses) = training.train_network_e2e(data, params)

# SAVE
output_directory = "./end2end"
case_name = "lr{}_weights{}-{}-{}-{}".format(params["learning_rate"], params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"], params["loss_weight_sindy_reg"])
# total losses
with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump((loss, losses), pickle_file)
# sindy_coeffs
with open(output_directory + '/sindy/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump(sindy_coeffs, pickle_file)
# vae model
torch.save(vae.state_dict(), output_directory + '/autoencoder/' + case_name + ".pth")
print(f"Saved model and losses as {case_name}")
