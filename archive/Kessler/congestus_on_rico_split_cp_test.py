# script for testing how accurate the conformal predictions actually are in terms of coverage and all that!

import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(
    description="Test accuracy of conformal prediction coverages, serially"
)

print("loading testing data")
# Load congestus dataset as training set
congestus_dset = np.load("data/congestus_test_processed.npy")
st_congestus = congestus_dset[..., 1:]

# Load rico dataset as calibration + test set
rico_dset = np.load("data/rico_test_processed.npy")
times_rico = rico_dset[..., 0]
st_rico = rico_dset[..., 1:]

# Reduce number of timesteps so they match in each dataset
n_t = min(st_congestus.shape[1], st_rico.shape[1])
st_congestus = st_congestus[:, :n_t]
times_rico = times_rico[:, :n_t]
st_rico = st_rico[:, :n_t]

# Normalize both rico and congestus datasets
# number of scalar‐elements in each set
N1 = st_congestus.shape[0] * st_congestus.shape[1]
N2 = st_rico.shape[0] * st_rico.shape[1]

# per‐dataset mean over (0,1)
mean_1 = st_congestus.mean(axis=(0, 1))
mean_2 = st_rico.mean(axis=(0, 1))

# weighted combination to get the global mean
means = (mean_1 * N1 + mean_2 * N2) / (N1 + N2)
st_congestus /= means[None, None, :]
st_rico /= means[None, None, :]
del congestus_dset, rico_dset, st_congestus

# load all outputs from cp_SINDy.py/cp_SINDy_parallel.py
with open("data/congestus_on_rico_split/cp.pkl", "rb") as f:
    alphas, idx_test, lower, upper, rep = pickle.load(f)

# preallocate array to contain percent of data lying within each prediction band
percents = np.empty((len(alphas), st_rico.shape[1] - 1), dtype=float)

# get real test trajectories
test_states = st_rico[idx_test, 1:]
N = len(test_states)
# loop across alphas
for i, alpha in enumerate(alphas):
    print(f"testing for coverage 1-alpha={100*(1-alpha)}%")
    # test if it falls within the bands
    testing = (test_states >= lower[i]) & (test_states <= upper[i])
    # get fraction that fall within the bands
    counter = np.count_nonzero(testing, axis=0) / N
    print(
        f"Mean percent of real test trajectories that fall within: {100*np.mean(counter, axis=(0, 1))}"
    )
    print(
        f"Median percent of real test trajectories that fall within: {100*np.median(counter, axis=(0, 1))}"
    )
    print(
        f"Standard deviation of percent of real test trajectories that fall within: {100*np.std(counter, axis=(0, 1))}"
    )
