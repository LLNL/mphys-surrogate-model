# script for testing how accurate the conformal predictions actually are in terms of coverage and all that!

import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(
    description="Test accuracy of conformal prediction coverages, serially"
)

parser.add_argument(
    "-m", "--method", type=str, required=True, help="Name of the cp method to test"
)

args = parser.parse_args()
method = args.method

print("loading testing data")
# load congestus dataset as training set
train_dset = np.load("data/congestus_test_processed.npy")
st_train = train_dset[..., 1:]

# load rico dataset as test set
test_dset = np.load("data/rico_test_processed.npy")
times_test = test_dset[..., 0]
st_test = test_dset[..., 1:]

# reduce number of timesteps so they match in test set only
n_t = min(st_train.shape[1], st_test.shape[1])
times_test = times_test[:, :n_t]
st_test = st_test[:, :n_t]

# normalize test set
# number of scalar‐elements in each set
N1 = st_train.shape[0] * st_train.shape[1]
N2 = st_test.shape[0] * st_test.shape[1]

# per‐dataset mean over (0,1)
mean_1 = st_train.mean(axis=(0, 1))
mean_2 = st_test.mean(axis=(0, 1))
del train_dset, test_dset, st_train

# weighted combination to get the global mean
means = (mean_1 * N1 + mean_2 * N2) / (N1 + N2)
st_test /= means[None, None, :]

# load all outputs from cp_SINDy.py/cp_SINDy_parallel.py
with open("data/congestus_on_rico/cp_" + str(method) + ".pkl", "rb") as f:
    alphas, lower, upper, rep = pickle.load(f)

# preallocate array to contain percent of data lying within each prediction band
percents = np.empty((len(alphas), st_test.shape[1] - 1), dtype=float)

# get real test trajectories
test_states = st_test[:, 1:]
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
