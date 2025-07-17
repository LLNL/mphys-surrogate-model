# script takes in data filepath and does UQ using b(r)agging E-SINDy 
# data should be an .npy file consisting of a list of trajectories, each of an array of size (# of timesteps, # of state space dimensions +1)
# for each trajectory, the array should have the times in the first column and the state-space variable values in the remaining columns

import numpy as np
import pysindy as ps

# load .npy data file with the name indicated by file path
file_path = input()
dset = np.load(file_path)

# get list of trajectories for times and states
times = list(dset[:, :, 0])
states = dset[:, :, 1:]
means  = states.mean(axis=(0,1))
states /= means[None, None, :]
states = list(states) # normalize q_c and q_r variables by mean

from pysindy.optimizers import STLSQ

# a = 0.001 # threshold for autoconversion in kg water/kg air
# a *= 0.001225 / mean # convert tolerance to volume water/volume air, normalized 
# print(a)
# a = 2.340384636957035

# create custom fractional exponent library
library_functions = []
library_function_names = []
library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.875) * np.sign(y)) # for rate of accretion
library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.875)+')')
library_functions.append(lambda x, y: y * np.power(np.abs(x), 0.875) * np.sign(x)) # for rate of accretion
library_function_names.append(lambda x, y: y + ' ' + x+'^('+str(0.875)+')')
# turn these off below if you are doing dealing with a no-advection model
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.525) * np.sign(y)) # for rate of evaporation
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.525)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.7296) * np.sign(y)) # for ventilation factor term
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.7296)+')')
# library_functions.append(lambda x, y: x * np.power(np.abs(y), 0.1346) * np.sign(y)) # for terminal fall velocity
# library_function_names.append(lambda x, y: x + ' ' + y+'^('+str(0.1346)+')')
library_functions.append(lambda x: np.maximum(0, x - 2.340384636957035)) # autoconversion rate
library_function_names.append(lambda x: 'max(0,'+x+'-a)')
custom_library = ps.CustomLibrary(library_functions=library_functions, 
                                  function_names=library_function_names)
feature_library = ps.PolynomialLibrary(degree=3, include_bias=False) + custom_library
# feature_library += custom_library * custom_library

# defining training for SINDy model
# uses STLSQ by default
threshold = 5e-5 # sparsity parameter
optimizer = STLSQ(threshold=threshold)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

# compute derivatives separately for each initial condition
# DON'T use SINDy predictions for this
derivatives = times # using times as it has the same shape
# compute derivatives separately for each initial condition
for i, t in enumerate(times):
    derivatives[i] = ps.differentiation.FiniteDifference()._differentiate(states[i], times[i])
derivatives = np.concatenate(derivatives) # flatten/concatenate derivatives list
states = np.concatenate(states) # flatten all samples
N = states.shape[1] # number of variables

# run loop for b(r)agging E-SINDy
q = int(np.round(0.05 * len(states))) # number of data bootstraps
fit_model = model.fit(states, x_dot=derivatives) # run on whole data first to get # of output features and preallocate
bootstraps = np.zeros((q, N, model.n_output_features_)) # preallocate coefficient samples for each bootstrap
for i in range(q): # sample with replacement q times
    indices = np.random.randint(0, len(states), size=len(states)) # indices to select for bootstrap
    model.fit(states[indices], x_dot=derivatives[indices]) # solve model on resampled dataset 
    bootstraps[i] = model.coefficients() # save model coefficients for sample
    print('Finished bootstrap', i)

features = model.get_feature_names()

import pickle
# save the trained coefficient samples and feature names using pickle
with open(file_path.removesuffix("processed.npy")+'esindy.pkl', 'wb') as f:
    pickle.dump((bootstraps, features), f)