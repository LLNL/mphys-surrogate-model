# script takes in data filepath and trains it on a SINDy model
# data should be an .npy file consisting of a list of trajectories, each of an array of size (# of timesteps, # of state space dimensions +1)
# for each trajectory, the array should have the times in the first column and the state-space variable values in the remaining columns

import numpy as np
import pysindy as ps
import argparse

parser = argparse.ArgumentParser(description="SINDy with custom threshold")
parser.add_argument("file", help="Path to *_processed.npy dataset")
# good sparsity parameter for box: 0.0005
# good sparsity parameter for 100m_full: 0.0001
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=1e-4,  # default threshold
    help=f"Sparsity threshold for STLSQ (default=1e-4)",
)

args = parser.parse_args()

# Override the module‐level THRESHOLD
file_path = args.file
threshold = args.threshold

# load .npy data file with the name indicated by file path
dset = np.load(file_path)

# get list of trajectories for times and states
times = dset[..., 0]
states = dset[..., 1:]
var_means = states.mean(axis=(0, 1))
states = states / var_means[None, None, :]  # normalize each variable by its own mean

from pysindy.optimizers import STLSQ
from sklearn.linear_model import (
    RidgeCV,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    LinearRegression,
)

# a = 0.001 # threshold for autoconversion in kg water/kg air
# a *= 0.001225 / mean # convert tolerance to volume water/volume air, normalized
# print(a)
# a = 2.340384636957035

# create custom fractional exponent library
library_functions = []
library_function_names = []
library_functions.append(
    lambda x, y: x * np.power(np.abs(y), 0.875) * np.sign(y)
)  # for rate of accretion
library_function_names.append(lambda x, y: x + " " + y + "^(" + str(0.875) + ")")
library_functions.append(
    lambda x, y: y * np.power(np.abs(x), 0.875) * np.sign(x)
)  # for rate of accretion
library_function_names.append(lambda x, y: y + " " + x + "^(" + str(0.875) + ")")
# turn these off below if you are doing dealing with a no-advection model
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.525) * np.sign(y)) # for rate of evaporation
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.525)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.7296) * np.sign(y)) # for ventilation factor term
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.7296)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.1346) * np.sign(y)) # for terminal fall velocity
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.1346)+')')
library_functions.append(
    lambda x: np.maximum(0, x - 2.340384636957035)
)  # autoconversion rate
library_function_names.append(lambda x: "max(0," + x + "-a)")
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False) + custom_library
# feature_library += custom_library * custom_library

# train and optimize SINDy model

# optimizer = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1))
optimizer = STLSQ(threshold=threshold)
# optimizer = MultiTaskLassoCV()
# optimizer = MultiTaskElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=500)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
fit_model = model.fit(list(states), t=list(times), multiple_trajectories=True)
print("Feature names:\n", model.get_feature_names())

print(fit_model.print())
print(model.coefficients())

import dill

# save the trained SINDy model using dill
with open(file_path.removesuffix("processed.npy") + "sindy_model.pkl", "wb") as f:
    dill.dump(model, f)
