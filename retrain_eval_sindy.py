import numpy as np
import matplotlib.pyplot as plt
import data_utils as du
import xarray as xr
import torch
import pickle as pkl
import glob
import training
import models
import pysindy as ps
import random
import matplotlib as mpl
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
import seaborn as sns

output_directory = "./hyperparam_e2e"
device = "cpu"
loss = []
losses = []
filenames = []
sindy_v0 = []
vaes = []

for filename in glob.glob(output_directory + "/losses/*.pkl"):
    prefix = output_directory + "/losses/"
    fn = filename[len(prefix):]
    filenames.append(fn)
    with open(filename, 'rb') as pickle_file:
        (loss_i, losses_i) = pkl.load(pickle_file)
        loss.append(loss_i)
        losses.append(losses_i)
    with open(output_directory + "/sindy/" + fn[:-4] + ".pkl", 'rb') as pickle_file:
        sindy_i = pkl.load(pickle_file)
        sindy_v0.append(sindy_i)

    latent_dim = int(fn[6])
    print(latent_dim)

    vae = models.CNNAutoEncoder(n_channels=1, n_bins=63, n_latent=latent_dim)
    device = torch.device(device)
    vae.load_state_dict(torch.load(output_directory + "/autoencoder/" + fn[:-4] + ".pth", map_location=device))