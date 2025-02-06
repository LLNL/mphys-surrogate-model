import sys
import numpy as np
import data_utils as du
from scipy.stats import wasserstein_distance
import xarray as xr
import torch
import pickle as pkl
import training
import uuid
from scipy.stats import qmc

train_models = True
eval_models = False
filepath = "box64.nc"

params_rng = {}
params_rng['latent_dim'] = (2, 4)
params_rng['poly_order'] = (1, 3)
params_rng["pretraining_epochs"] = (0, 1)  # boolean
params_rng["training_epochs"] = (0, 1)      # boolean
params_rng["refinement_epochs"] = (0, 1)    # boolean
params_rng['loss_weight_sindy_z'] = (-4, 2) # logscale
params_rng['loss_weight_sindy_x'] = (-4, 2) # logscale
params_rng['loss_weight_sindy_reg'] = (-4, 2) # logscale
params_rng["learning_rate"] = (-4, -1) # logscale

n_samples = 9

def get_psd_evolution(x0, coeff, vae, t_sim, zlim, p_order=2):
    z0 = vae.encoder(torch.Tensor(x0.reshape(1, -1))).detach().numpy()[0][0]
    z_sim = du.sindy_simulate(z0, t_sim, torch.tensor(coeff).float(), p_order, zlim)
    x_sim = np.zeros((len(T), len(x0)))
    for (j, t) in enumerate(t_sim):
        x_sim[j, :] = vae.decoder(torch.Tensor(z_sim[j])).detach().numpy()[0][0]

    return x_sim

def compute_sim_error(X_train, X_test, vae, coeff, T, params, outdir, case_name):
    ztr_encoded = vae.encoder(torch.tensor(X_train).reshape(-1, 1, params["input_dim"]))
    ztr_encoded = ztr_encoded.reshape(X_train.shape[0], -1, params["latent_dim"]).detach().numpy()

    # retrain sindy
    coeff = retrain_sindy(ztr_encoded, T, outdir, case_name)

    zlim = np.zeros((params["latent_dim"], 2))
    for il in range(params["latent_dim"]):
        zlim[il][0] = ztr_encoded[:, :, il].min()
        zlim[il][1] = ztr_encoded[:, :, il].max()

    w1_dist = np.zeros((X_test.shape[0], len(T)))
    l2_err = np.zeros_like(w1_dist)
    for i in range(len(X_test)):
        x_0 = X_test[i, 0]
        x_sim = get_psd_evolution(x_0, coeff, vae, T, zlim, p_order=params["poly_order"])
        for (j, t) in enumerate(T):
            w1_dist[i, j] = wasserstein_distance(X_test[i, j], x_sim[j])
            l2_err[i, j] = np.linalg.norm(X_test[id, j] - x_sim[j])

    np.savez(outdir + "/metrics/" + case_name + ".pkl",
            l2_err=l2_err,
            w1_dist=w1_dist)

    return np.sum(w1_dist)

def retrain_sindy(ztr_encoded, T, output_dir, case_name):
    import pysindy as ps
    optimizer = ps.SR3(
        threshold=1e-1, thresholder="l0", max_iter=1000, normalize_columns=False, tol=1e-1
    )
    sindy_model = ps.SINDy(
        optimizer=optimizer,
        feature_library=ps.PolynomialLibrary(2)
    )
    sindy_model.fit(ztr_encoded, t=T)

    with open(output_dir + '/sindy_retrain/' + case_name + '.pkl', 'wb') as pickle_file:
        pkl.dump(optimizer.coef_, pickle_file)

    return optimizer.coef_


def train_model(ds_all, params, num_eval=2, split=(80, 10, 10), output_directory = "./hyperparam_e2e"):
    for _ in range(num_eval):
        (data, norms, data_loaders) = du.create_e2e_dataloader(ds_all, cnn=params["CNN"], batch_size=params["batch_size"], tvt_split=split)
        (train_data, val_data, test_data) = data_loaders
        (X, DX, T) = data

        (vae, sindy_coeffs, loss, losses, val_loss, val_losses) = training.train_network_e2e(train_data, params, val_dataloader=val_data, device=params["device"])

        if params["CNN"]:
            prefix = "CNN"
        else:
            prefix = "FFNN"
        case_name = prefix + "_nl{}_order{}_tr{}-{}-{}_lr{}_weights{}-{}-{}-{}_{}".format(
            params["input_dim"],
            params["poly_order"],
            params["pretraining_epochs"], params["training_epochs"], params["refinement_epochs"],
            params["learning_rate"],
            params["loss_weight_recon"], params["loss_weight_sindy_z"], params["loss_weight_sindy_x"],
            params["loss_weight_sindy_reg"],
            uuid.uuid4().hex)

        with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
            pkl.dump((loss, losses), pickle_file)
        # sindy_coeffs
        with open(output_directory + '/sindy/' + case_name + '.pkl', 'wb') as pickle_file:
            pkl.dump(sindy_coeffs, pickle_file)
        # vae model
        torch.save(vae.state_dict(), output_directory + '/autoencoder/' + case_name + ".pth")
        print(f"Saved model and losses as {case_name}")


# MAIN #
device = torch.device(
	"cuda"
	if torch.cuda.is_available()
	else "cpu"
    )
torch.backends.cudnn.benchmark = (
	True
    )
print(f"Using {device} device")

# Set up the param set
ds_all = xr.open_dataset(filepath)
params = {}
params['input_dim'] = len(ds_all['mass_bin'])
params["n_runs"] = len(ds_all['run'])
params["n_time"] = len(ds_all['time'])
params["device"] = device
params["batch_size"] = 400
params["CNN"] = True
params["patience"] = 50
params['loss_weight_recon'] = 1e0

# get the LHS and scale the parameters as integers
lhs_ranges = list(params_rng.values())
sampler = qmc.LatinHypercube(d = len(lhs_ranges))
samples_unit = sampler.random(n=n_samples)
lb = [g[0] for g in lhs_ranges]
ub = [g[1] for g in lhs_ranges]
samples_int = np.round(qmc.scale(samples_unit, lb, ub)).astype(int)

for sample in samples_int:
    params['loss_weight_recon'] = 1e0
    for (i, key) in enumerate(params_rng.keys()):
        if i <= 1:
            params[key] = sample[i]
        elif i <= 4:
            params[key] = 1 + int(sample[i] * 999)
        else:
            params[key] = 1e1**(sample[i])

    # rescale the weights
    max_weight = 1.0
    for key in list(params.keys()):
        if key.startswith("loss_weight"):
            if params[key] > max_weight:
                max_weight = params[key]
    for key in list(params.keys()):
        if key.startswith("loss_weight"):
            params[key] /= max_weight

    print(params)
    train_model(ds_all, params)