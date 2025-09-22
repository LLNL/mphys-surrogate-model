"""
Script for testing how accurate the conformal predictions actually are in terms of coverage
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, "../.."))
sys.path.append(parent_directory)  # parent_directory = ~/mphys-surrogate-model

from src import data_utils as du

params = {
    "random_seed": 1952,
    "latent_dim": 3,
}

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-s",
    "--subset",
    type=str,
    required=True,
    choices=["decoder", "latent", "full", "mass"],
    help="which subset of the network you want to test coverage for conformal predictions on: decoder, latent, full, or mass"
    "-decoder: the reconstruction/autoencoder only"
    "-latent: the dynamics in the latent space only"
    "-full: the entire network architecture (reconstruction+dynamics)"
    "-mass: the (normalized) mass as a function of time",
)
parser.add_argument(
    "-a",
    "--model",
    type=str,
    required=True,
    choices=["AR", "NNdzdt", "SINDy"],
    help="the dynamic model/architecture used (required): AR, NNdzdt, or SINDy",
)
parser.add_argument(
    "-p",
    "--p",
    type=int,
    default=20,
    help="The p indicates what *percent* you want to dedicate out of the full data for calibration."
    "Default is 20%, in which case p=20.",
)
args = parser.parse_args()

calib_size = args.p

# load results from conformal predictions
cp_results_file = (
    os.path.basename(os.path.normpath(args.data_name))
    + "_split"
    + str(calib_size)
    + ".pkl"
)
pickle_path = os.path.join(
    parent_directory,
    "UQ",
    "conformal",
    "results",
    "ae_" + args.model,
    cp_results_file,
)
with open(pickle_path, "rb") as f:
    alphas, _, DSD_bands, m_bands, latent_dict = pickle.load(f)

# load data
outputs = du.open_mass_dataset(
    name=args.data_name,
    data_dir=Path(parent_directory) / "data",
    sample_time=None,
    test_size=1 - 0.01 * calib_size,
    random_state=params["random_seed"],
)

subset = args.subset

# if we are testing conformal predictions on the latent space, we need to load the model
if subset == "latent":
    import torch

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "AR":
        from training_scripts import train_ae_ar as train

        params.update(
            {
                "n_lag": 1,
                "layer_size": [141, 154, 40],
            }
        )

        model = train.AEAutoregressor(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            n_lag=params["n_lag"],
            layer_size=params["layer_size"],
        )
        optimal_path = os.path.join(
            "results",
            "Optuna",
            "ERF Dataset",
            "AE-AR_2025-09-18T10:18:57_PostARBugfix",
            "erf_FFNN_latent3_order(141, 154, 40)_tr1000_lr0.0030348411572892766_bs8_weights0.12789450188986579-1.1729901013704414_a0f49326688d4e69bcc0e9a78da3c870",
        )
        ae_ar_checkpoint = torch.load(
            os.path.join(
                optimal_path,
                "erf_FFNN_latent3_order(141, 154, 40)_tr1000_lr0.0030348411572892766_bs8_weights0.12789450188986579-1.1729901013704414_a0f49326688d4e69bcc0e9a78da3c870.pth",
            ),
            weights_only=True,
        )
        model.load_state_dict(ae_ar_checkpoint)
    if args.model == "NNdzdt":
        from training_scripts import train_ae_NNdzdt as train

        params.update(
            {
                "layer_size": (42, 36, 46),
            }
        )

        model = train.AENNdzdt(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            layer_size=params["layer_size"],
        )
        optimal_path = os.path.join(
            "results",
            "Optuna",
            "ERF Dataset",
            "NNdzdt_2025-07-20T23:31:20_3a400c596947422389559813cd41dfe6",
            "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3",
        )
        ae_NNdzdt_checkpoint = torch.load(
            os.path.join(
                optimal_path,
                "erf_FFNN_latent3_layers(42, 36, 46)_tr1000_lr0.00314227212817401_bs4_weights1.0-599.504638671875-59950.4609375_ecb1da0eabf9423ab03bed5ad82f43a3.pth",
            ),
            weights_only=True,
        )
        model.load_state_dict(ae_NNdzdt_checkpoint)
    if args.model == "SINDy":
        from training_scripts import train_ae_sindy as train

        params.update(
            {
                "poly_order": 2,
            }
        )

        model = train.AESINDy(
            n_channels=1,
            n_bins=outputs["n_bins"],
            n_latent=params["latent_dim"],
            poly_order=params["poly_order"],
        )
        optimal_path = os.path.join(
            "results",
            "Optuna",
            "ERF Dataset",
            "AE-SINDy_LimParams",
            "erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc",
        )
        ae_sindy_checkpoint = torch.load(
            os.path.join(
                optimal_path,
                "erf_FFNN_latent3_order2_tr1000_lr0.004204813405972317_bs25_weights1.0-561.064697265625-56106.47265625_46d657b7ac094414a37843315fdeebbc.pth",
            ),
            weights_only=True,
        )
        model.load_state_dict(ae_sindy_checkpoint)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        z_enc_test = (
            model.encoder(
                torch.tensor(outputs["x_test"], device=device, dtype=torch.float32)
            )
            .detach()
            .cpu()
            .numpy()
        )

# get prediction bands
if subset == "decoder":
    lower = DSD_bands[0][0]
    upper = DSD_bands[1][0]
    rep = DSD_bands[2][0]
elif subset == "latent":
    Sigma_inv = latent_dict["Sigma_inv"]
    taus = latent_dict["taus"]
    r_test = latent_dict["z_enc_test_pred"] - z_enc_test
    mu = latent_dict.get("mu", None)
    if mu is not None:
        r_test = r_test - mu[None, :, :]
elif subset == "full":
    lower = DSD_bands[0][1]
    upper = DSD_bands[1][1]
    rep = DSD_bands[2][1]
elif subset == "mass":
    lower = m_bands[0]
    upper = m_bands[1]
    rep = m_bands[2]
else:
    raise KeyError("Subset of architecture indicated (via -s) has not been implemented")

"""
Now we test the conformal prediction coverage!
"""
N = len(outputs["x_test"])  # number of samples/initial conditions
if subset == "latent":
    scores_test = np.einsum("mnd,ndd,mnd->mn", r_test, Sigma_inv, r_test)
# loop across alphas
for i, alpha in enumerate(alphas):
    print(f"testing for coverage 1-alpha={100*(1-alpha)}%")
    # test if it falls within the prediction set
    if subset == "latent":
        testing = scores_test <= taus[i]
    elif subset == "mass":
        testing = (outputs["m_test"] >= lower[i]) & (outputs["m_test"] <= upper[i])
    else:
        testing = (outputs["x_test"] >= lower[i]) & (outputs["x_test"] <= upper[i])
    # get fraction that fall within the prediction set
    counter = np.count_nonzero(testing, axis=0) / N
    if (subset == "mass") or (subset == "latent"):
        print(
            f"Mean percent of real test trajectories that fall within: {100*np.mean(counter)}"
        )
        print(
            f"Median percent of real test trajectories that fall within: {100*np.median(counter)}"
        )
        print(
            f"Standard deviation of percent of real test trajectories that fall within: {100*np.std(counter)}"
        )
    else:
        print(
            f"Mean percent of real test trajectories that fall within: {100*np.mean(counter, axis=(0, 1))}"
        )
        print(
            f"Median percent of real test trajectories that fall within: {100*np.median(counter, axis=(0, 1))}"
        )
        print(
            f"Standard deviation of percent of real test trajectories that fall within: {100*np.std(counter, axis=(0, 1))}"
        )
