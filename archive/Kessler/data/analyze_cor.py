""" 
This script loads spatial correlations between each gridbox and the others and plots some of them.
-
"""

import numpy as np
import argparse

# get data_name and threshold for significance test
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.01,
    help="the p-value threshold for significance",
)
args = parser.parse_args()

import pickle

# load pickle file containing neighboring indices as needed
with open(args.data_name + "_neighbors.pkl", "rb") as f:
    neighbor_list = pickle.load(f)
# load pickle file containing cross-portmanteau test p-values amongst different neighbors
with open(args.data_name + "_CCF_test.pkl", "rb") as f:
    CCFs_pvals = pickle.load(f)
# load pickle file containing linear correlation test p-values amongst different neighbors
with open(args.data_name + "_linear_test.pkl", "rb") as f:
    linear_pvals = pickle.load(f)

# for now, just go through all of the p-values and check what proportion of them pass certain significance thresholds
CCFs_count = linear_count = 0  # initialize counter for number of significant p-values
CCFs_total = linear_total = 0  # initialize counter for total length of array
for i in range(len(neighbor_list)):
    CCFs_count += sum(1 for x in CCFs_pvals[i] if x[-1] < args.threshold)
    linear_count += sum(1 for x in linear_pvals[i] if x < args.threshold)
    CCFs_total += len(CCFs_pvals[i])
    linear_total += len(linear_pvals)
print(CCFs_count / CCFs_total)
print(linear_count / linear_total)
