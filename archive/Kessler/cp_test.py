# script for testing how accurate the conformal predictions actually are in terms of coverage and all that!

import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(
    description="Test accuracy of conformal prediction coverages, serially"
)

parser.add_argument("file_path", help="Name of data for cp results (e.g., rico_test)")

parser.add_argument(
    "-m", "--method", type=str, required=True, help="Name of the cp method to test"
)

args = parser.parse_args()
file_path = "data/" + args.file_path
method = args.method

file_path_processed = file_path + "_processed.npy"
cp_file_path = file_path + "/cp_" + method + ".pkl"

print("loading data")
# load dataset
dset = np.load(file_path_processed)
times = dset[..., 0]
states = dset[..., 1:]
var_means = states.mean(axis=(0, 1))
states = states / var_means[None, None, :]  # normalize each variable by its own mean

# load all outputs from cp_SINDy.py/cp_SINDy_parallel.py
with open(cp_file_path, "rb") as f:
    alphas, idx_test, lower, upper, rep = pickle.load(f)

# preallocate array to contain percent of data lying within each prediction band
percents = np.empty((len(alphas), len(times) - 1), dtype=float)

# get real test trajectories
test_states = states[idx_test, 1:]
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
