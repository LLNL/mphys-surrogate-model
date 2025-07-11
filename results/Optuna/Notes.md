This folder has the results for Optuna trials. Each sub folder is split by
model type, and contains a "true" run of the model with the parameters
found in the optuna run. Optuna is trying to minimize the mean wasserstein 
distance for the entire training set. Current tests are with box data.

We can get decent results with a second degree polynomial for AE-SINDy,
see AE-SINDy_2025-07-11T12/08/26_d89a4eb0fb4945028106fcf78349108c folder.
More work is needed for SINDy and Optuna though. 