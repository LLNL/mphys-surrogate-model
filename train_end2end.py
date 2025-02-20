import sys
import data_utils as du
import xarray as xr
import torch
import pickle as pkl
import training
import uuid

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
params['poly_order'] = "BB" #int(sys.argv[1])
params["tracemalloc"] = True
params["pretraining_epochs"] = 1
params["training_epochs"] = 1
params["refinement_epochs"] = 0
params['loss_weight_recon'] = 10.0 #float(sys.argv[2])
params['loss_weight_sindy_z'] = 1.0
params['loss_weight_sindy_x'] = 1.0
params['loss_weight_sindy_reg'] = 0.0 #float(sys.argv[3])
params["learning_rate"] = 1e-3
params["CNN"] = True #if sys.argv[4] == "CNN" else False
params["patience"] = 50

print(params)

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

if params["poly_order"] == "BB":
    (vae, dzdt, loss, losses, val_loss, val_losses) = training.train_network_e2e_bb(train_data, params,
                                                                                         val_dataloader=val_data,
                                                                                         device=params[
                                                                                             "device"])  # , X=X, T=T)
else:
    (vae, sindy_coeffs, loss, losses, val_loss, val_losses) = training.train_network_e2e(train_data, params, val_dataloader=val_data, device=params["device"])#, X=X, T=T)

if params["tracemalloc"]:
    # displaying the memory
    print("Memory Usage: Current (MB) | Peak (MB)")
    print([tracemalloc.get_traced_memory()[i] / 1024**2 for i in range(2)])

    # stopping the library
    tracemalloc.stop()


# SAVE
output_directory = "./end2end_bb" #"./end2end"
# if params["CNN"]:
#     prefix = "CNN"
# else:
#     prefix = "FFNN"
# case_name = prefix + "_order{}_{}-{}-{}_lr{}_weights{}-{}-{}-{}_{}".format(
#         params["poly_order"],
#         params["pretraining_epochs"], params["training_epochs"], params["refinement_epochs"], params["learning_rate"],
#         params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"], params["loss_weight_sindy_reg"],
#         uuid.uuid4().hex)
case_name = "CNN_BB_tr{}-{}-lr{}_weights{}-{}-{}".format(params["pretraining_epochs"], params["training_epochs"], params["learning_rate"], params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"])

# total losses
with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump((loss, losses), pickle_file)
# sindy_coeffs
if params["poly_order"] == "BB":
    torch.save(dzdt.state_dict(), output_directory + '/dzdt/' + case_name + ".pth")
else:
    with open(output_directory + '/sindy/' + case_name + '.pkl', 'wb') as pickle_file:
        pkl.dump(sindy_coeffs, pickle_file)
# vae model
torch.save(vae.state_dict(), output_directory + '/autoencoder/' + case_name + ".pth")
print(f"Saved model and losses as {case_name}")
