# this script generates q_c and q_r for a given DSD dataset in mass
# for cases where the data are in mass, not volumes

import xarray as xr
import numpy as np

file_path = input()
dset = xr.open_dataset(file_path, decode_times=False)
DSDs = np.transpose(dset['dmdlnr'].values, (2,1,0)) # extract mass density vs. ln(r)
# extract values for ln(r)
bins = np.append(dset['rbin_l'].values, dset['rbin_r'].values[-1])
log_bins = np.log(bins)
dlog_bins = np.diff(log_bins) # bin sizes for integration
t = dset['t'][:]

# index threshold for dividing q_c vs. q_r
threshold_val = np.log(4.5 * 10 ** (-5))
# find index for mass threshold
threshold_ind = next(i for i,r in enumerate(log_bins) if r > threshold_val)

N = len(DSDs) # number of gridboxes

tol = 10 ** (-5) # tolerance for excluding empty gridboxes, in kg water/kg air
q_c = q_r = total = np.empty(len(t)) # preallocate q_c and q_r

from scipy.integrate import trapezoid

# iterate through each gridbox using list comprehension. get a list of trajectories
data_all = [] # preallocate list of trajectories
for i in range(N):
    # compute total for the current grid box using the composite trapezoidal rule
    total = np.dot(DSDs[i], dlog_bins)
    # only process if the condition (at least one value over tol) is met
    if total.max() > tol:
        # compute q_c only for the threshold portion.
        q_c = np.dot(DSDs[i][:, :threshold_ind], dlog_bins[:threshold_ind])
        # stack t, q_c, and (total - q_c) into a single array
        gridbox_data = np.column_stack((t, q_c, total - q_c))
        # append the result to the list
        data_all.append(gridbox_data)
# final array is of size (number of nontrivial gridboxes, number of timesteps, 3)
np.save(file_path.removesuffix(".nc")+'_processed', data_all) # save final array