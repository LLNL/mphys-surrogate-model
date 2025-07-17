'''
Script takes in data filepath and does UQ 
Data should be an .npy file consisting of a list of trajectories, each of an array of size (# of timesteps, # of state space dimensions +1)
For each trajectory, the array should have the times in the first column and the state-space variable values in the remaining columns
In this script, ALL time series in each gridbox must be of the same length and correspond to the same times (ideally).
'''

import numpy as np
import pysindy as ps
import argparse

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument("-t", "--test_size", type=float, default=0.2,
                    help="testing set proportion, must be between 0 and 1, default is 0.2")
parser.add_argument("-m", "--method", default="cv+5",
                    help="conformal predictions method to use (jackknife, split[p], full, cv+[k]), default is cv+5. \
                        The p in split indicates what *percent* you want to dedicate out of the training data for calibration, while\
                            the k in cv+ indicates how many models to train for cross-validation folds.")
parser.add_argument("-a", "--alpha", nargs="+", type=float, default=0.1,
                    help="miscoverage rate(s), must be between 0 and 1, default is 0.1")
args = parser.parse_args()

test_size = args.test_size
method = args.method
# isolate p or k in the situation where you're using split conformal or cv+
if method[:5] == "split":
    p = float(method[5:])
    if (p <= 0) | (p >= 100):
        raise ValueError("Calibration size for split must be a percent strictly between 0 and 100.")
    method = "split"
if method[:3] == "cv+":
    k = int(method[3:])
    method = "cv+"
if method not in ["jackknife", "split", "full", "cv+"]: # raise error if method is not one of the list above
    raise ValueError("Invalid conformal predictions method specified.")

alphas = args.alpha if isinstance(args.alpha, (list, tuple)) else [args.alpha]

if (test_size <= 0) | (test_size >= 1):
    raise ValueError("Test set proportion must be between 0 and 1")
if any(a <= 0 or a >= 1 for a in alphas):
    raise ValueError("Coverage rate (alpha) must be between 0 and 1 for all values")

def one_sided_quantiles(residuals, alpha_lows, alpha_ups):
    """
    Compute all lower- and upper-tail quantiles in one shot.

    residuals : array_like, shape (N, T, D)
    alpha_lows: list of alpha/2 levels (e.g. [0.125, 0.025] for 75% & 95%)
    alpha_ups : same as alpha_lows
    Returns
    -------
    q_low, q_high : arrays of shape (len(alpha_lows), T, D)
    """
    import numpy as _np

    # 1) build the full list of levels
    lows  = _np.array(alpha_lows)
    ups   = 1.0 - _np.array(alpha_ups)
    all_q = _np.concatenate([lows, ups])       # e.g. [0.125, 0.025, 0.875, 0.975]

    # 2) sort levels and remember how to invert
    sort_idx = _np.argsort(all_q)
    q_sorted = all_q[sort_idx]

    # 3) single quantile call
    qs = _np.quantile(residuals, q_sorted, axis=0)

    # 4) invert the sort
    qs_unsorted = _np.empty_like(qs)
    qs_unsorted[sort_idx] = qs

    # 5) split into lows / highs
    m = len(alpha_lows)
    q_low  = qs_unsorted[:m]
    q_high = qs_unsorted[m:]
    return q_low, q_high

# load .npy data file with the name indicated by file path
data_name = args.data_name
dset = np.load(data_name)

# get list of trajectories for times and states
times = np.array(dset[:, :, 0])
states = np.array(dset[:, :, 1:])
var_means   = states.mean(axis=(0,1))            
states      = states / var_means[None,None,:] # normalize each variable by its own mean
del dset

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
threshold = 0.0001 # sparsity parameter
optimizer = STLSQ(threshold=threshold)
model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

# define integration functions
def fun(model, t, x):
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    dxdt = model.predict(x.reshape(1, -1)) 
    return dxdt[0]

def fun_damped(model, t, x):
    # epsilon scaling for fifth-order damping term
    # ratio of this term to the smallest yet non-zero coefficient 
    eps = 0.01 * 10 ** (-5)
    abs_coefs = np.abs(model.coefficients())
    final_eps = eps * np.min(abs_coefs[abs_coefs>0]).flatten() # actual coefficient value
    
    # reshape x to 2D as expected by predict, then return the derivative as 1D array.
    x_eval = x.reshape(1, -1)
    dxdt = model.predict(x_eval) - final_eps * (x_eval ** 5)
    return dxdt[0]

from scipy.integrate import solve_ivp 

def evolve_states(model, initial_states, times):
    # this function takes in samples of initial states and returns these at the times indicated 
    # initial_states: (n_traj, n_vars)
    # times: (n_traj, n_t)
    # returns: predicted_states: (n_traj, n_t, n_vars)
    
    # generate model data by running trained system of ODEs using BDF
    t = times[0] # times at each time (assume same time for all)
    n_traj = len(times)
    n_t = len(t)
    predicted_states = np.empty((n_traj, n_t, initial_states.shape[1]), dtype=float)
    for traj_no in range(n_traj): # loop through all trajectories/initial conditions
        # allocate storage: will become shape (n_t, 2)
        # because solve_ivp.y has shape (n_states, n_t)
        sol = np.empty((n_t, 2), dtype=float)

        # initial condition for this trajectory
        y0 = initial_states[traj_no]

        try:
            # 1) Try with the original SINDy RHS
            res = solve_ivp(
                fun     = lambda t, y: fun(model, t, y),
                t_span  = (t[0], t[-1]),
                y0      = y0,
                method  = "BDF",
                t_eval  = t,
                positive= True,      # enforce non‐negativity
            )
            if not res.success:
                # force an exception to trigger fallback
                raise RuntimeError(f"BDF failed (orig): {res.message}")

        except (RuntimeError, ValueError) as e:
            # 2) Fallback to the damped RHS
            res = solve_ivp(
                fun     = lambda t, y: fun_damped(model, t, y),
                t_span  = (t[0], t[-1]),
                y0      = y0,
                method  = "BDF",
                t_eval  = t,
                positive= True,
            )
            if not res.success:
                # escalate if both fail
                raise RuntimeError(
                    f"Trajectory {traj_no}: both original and damped BDF failed: {res.message}"
                ) from e

        # transpose to shape (n_t, n_states) so sol[i] is state @ t[i]
        sol[:] = res.y.T
        predicted_states[traj_no] = sol
    return predicted_states # return final list of predictions

# Finally, let's choose the conformal predictions method indicated as an input in the script!

from sklearn.model_selection import train_test_split

# make a global index array
n_all  = len(times)
idx_all = np.arange(n_all)            # [0,1,...,n_all-1]

# train/test split on indices
idx_train, idx_test = train_test_split(
    idx_all,
    test_size=test_size,
    random_state=1952
)

# split data into training and testing data (training includes calibration, FYI)
times_train  = times[idx_train]
times_test   = times[idx_test]
states_train = states[idx_train]
states_test  = states[idx_test]

from sklearn.base import clone

# split each alpha into two equal parts for lower/upper tails
alpha_lows = [a/2 for a in alphas]
alpha_ups  = alpha_lows.copy()

if method == 'jackknife':
    # 1) build leave-one-out (LOO) predictions
    n = states_train.shape[0]
    loo_preds = np.empty((n, states_train.shape[1]-1, states_train.shape[2]), float)  
    for i in range(n):
        m = clone(model)
        idx = np.arange(n) != i
        m.fit(list(states_train[idx]), t=list(times_train[idx]), multiple_trajectories=True)
        loo_preds[i] = evolve_states(model=m, initial_states=states_train[i,0].reshape(1,-1),
                                     times=times_train[i].reshape(1,-1))[0,1:]

    # 2) signed residuals on train set 
    res_signed = loo_preds - states_train[:,1:]  

    # 3) one-sided quantiles per alpha 
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups)  

    # 4) full-data fit 
    model_full = clone(model)
    model_full.fit(list(states_train), t=list(times_train), multiple_trajectories=True)
    preds = evolve_states(model=model_full,initial_states=states_test[:,0], times=times_test)[:,1:]       

    # 5) asymmetric bounds on test set 
    # reshape so we have (n_alphas, 1, T, D) - (1, n_test, T, D)
    lower = preds[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = preds[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = preds # central prediction
    
# if method == 'jackknife+':
#     # number of jackknife replicates & test trajectories
#     n = states_train.shape[0]
#     m = states_test.shape[0]
#     # time–state dims (drop the initial timepoint)
#     T = states_train.shape[1] - 1
#     D = states_train.shape[2]

#     # 1) Build leave‐one‐out predictions
#     preds = np.empty((n, m, T, D), dtype=float)
#     for i in range(n):
#         model_i = clone(model)
#         mask = np.arange(n) != i
#         model_i.fit(list(states_train[mask]), t=list(times_train[mask]), multiple_trajectories=True)
#         # evolve_states returns shape (m, T+1, D): drop t=0
#         preds[i] = evolve_states(model=model_i, initial_states=states_test[:, 0], times=times_test)[:, 1:]

#     # 2) Central (median) predictive trajectory
#     rep_traj = np.median(preds, axis=0) 

#     # 4) prepare true‐trajectories
#     y_true = states_test[:, 1:] # (m, T, D)
#     y_true = y_true[np.newaxis, :, :, :] # (1, m, T, D)

#     # 5) build prediction envelopes 
#     low_pts  = np.minimum(preds, y_true)
#     high_pts = np.maximum(preds, y_true)

#     # 6) lower‐bound: just the α_lows‐quantiles of low_pts
#     #    We pass alpha_lows for both low‐ and “unused” upper‐args,
#     #    then discard the second output.
#     lower, _ = one_sided_quantiles(
#         low_pts,
#         alpha_lows,      # q‐levels for the lower tail
#         alpha_lows       # dummy: we’ll ignore the high‐tail output
#     )

#     # 7) upper‐bound: the (1−α_ups)‐quantiles of high_pts.
#     #    So set both lists = [1−α for α in alpha_ups], then keep the first output.
#     hi_levels = [1.0 - a for a in alpha_ups]
#     upper, _ = one_sided_quantiles(
#         high_pts,
#         hi_levels,       # compute quantiles at 1−α
#         hi_levels       # dummy
#     )

    # return lower, upper, rep_traj
    
if method == 'full':
    # 1) fit & predict on training data 
    m = clone(model)
    m.fit(list(states_train), t=list(times_train), multiple_trajectories=True)
    pred_train = evolve_states(model=m, initial_states=states_train[:,0], times=times_train)[:,1:] 

    # 2) signed residuals on train set 
    res_signed = pred_train - states_train[:,1:]   

    # 3) one-sided quantiles per alpha
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups)      

    # 4) test predictions & bounds 
    pred_test = evolve_states(model=m, initial_states=states_test[:,0], times=times_test)[:,1:]   

    lower = pred_test[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = pred_test[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = pred_test
    
if method == 'split':
    # 1) split training and calibration sets
    n = states_train.shape[0]
    split = int(n*(1 - 0.01*p))
    states_fit, times_fit = states_train[:split], times_train[:split]
    states_cal, times_cal = states_train[split:], times_train[split:]

    # 2) fit on states_fit, predict on states_cal
    m = clone(model)
    m.fit(list(states_fit), t=list(times_fit), multiple_trajectories=True)
    pred_cal = evolve_states(model=m, initial_states=states_cal[:,0], times=times_cal)[:,1:]   

    # 3) signed residuals on calibration set
    res_signed = pred_cal - states_cal[:,1:]      

    # 4) one-sided quantiles per alpha
    q_low, q_high = one_sided_quantiles(res_signed, alpha_lows, alpha_ups) 

    # 5) test predictions & asymmetric bounds
    pred_test = evolve_states(model=m, initial_states=states_test[:,0],  times=times_test)[:,1:]   

    lower = pred_test[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = pred_test[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]
    rep_traj = pred_test
    
if method == "cv+":
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k, shuffle=True, random_state=1952)

    # collect residuals for each fold of shape (n_test, T, D)
    resid = []
    # collect per-fold predictions of shape (n_test, T, D)
    pred = []

    for train_idx, val_idx in kf.split(states_train):
        # 1) fit on fold‐k train
        m_k = clone(model)
        m_k.fit(
            list(states_train[train_idx]),
            t=list(times_train[train_idx]),
            multiple_trajectories=True
        )

        # 2) predict on validation (calibration) fold
        y0_val = states_train[val_idx, 0]  
        pred_val = evolve_states(
            model=m_k,
            initial_states=y0_val,
            times=times_train[val_idx]
        )  # returns shape (n_val, T_train, D)

        # 3) signed residuals on calibration fold
        resid.append(pred_val - states_train[val_idx])

        # 4) predict on the TEST set
        y0_test  = states_test[:, 0]        # (n_test, D)
        pred_test = evolve_states(
            model=m_k,
            initial_states=y0_test,
            times=times_test
        )  # (n_test, T_test, D)
        pred.append(pred_test)

    # 5) pool residuals and compute global quantiles
    resid_pool = np.concatenate(resid, axis=0)
    ql, qh = one_sided_quantiles(resid_pool, alpha_lows, alpha_ups)
    
    # 6) stack & intersect across folds
    rep_traj = np.median(np.stack(pred, axis=0), axis=0)   # intersect via median of K-fold predictions
    lower = rep_traj[np.newaxis, :, :, :] - q_high[:, np.newaxis, :, :]
    upper = rep_traj[np.newaxis, :, :, :] - q_low [:, np.newaxis, :, :]


# save alpha values, indices for test data, lower and upper interval values, and representative ("center") trajectories (in that order)

import pickle 

with open(data_name.removesuffix("processed.npy")+'cp_'+method+'.pkl', 'wb') as f:
    pickle.dump([alphas, idx_test, lower, upper, rep_traj], f)