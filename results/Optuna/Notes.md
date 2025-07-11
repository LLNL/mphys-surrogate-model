This folder has the results for Optuna trials. Each sub folder is split by
model type, and contains a "true" run of the model with the parameters
found in the optuna run. Optuna is trying to minimize the mean wasserstein 
distance for the entire training set. Current tests are with box data.