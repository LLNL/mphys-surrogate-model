# script to plot histograms for all coefficients from the result of eindy.py
# plots marginal distributions for each coefficient

import numpy as np
import matplotlib.pyplot as plt
import pickle

# import pickle file as inputed name
file_path = input()

# load coefficient samples from E-SINDy
with open(file_path, "rb") as f:
    bootstraps, features = pickle.load(f)
# transpose so we can loop through each coefficient
bootstraps = np.transpose(bootstraps, (1, 2, 0))


# use Freedman-Diaconis rule for number of bins
def freedman_diaconis_bins(data):
    """
    Freedman-Diaconis Rule:
    bin_width = 2 * IQR / n^(1/3) => bins = range / bin_width
    """
    data = np.asarray(data)
    if len(data) < 2:
        return 1  # not enough data
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1 / 3))

    if bin_width == 0:
        return int(np.ceil(np.sqrt(len(data))))

    data_range = data.max() - data.min()
    return int(np.ceil(data_range / bin_width))


# loop through each coefficient and plot marginal distribution
for i in range(bootstraps.shape[0]):  # loop through each variable i
    for j in range(bootstraps.shape[1]):  # coefficient j for ODE for variable i
        # get data
        data = bootstraps[i, j]

        # compute number of bins needed
        bins_fd = freedman_diaconis_bins(data)

        # create histogram
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins_fd, color="skyblue", edgecolor="black", density=True)
        plt.title(
            "Variable " + str(i) + ", feature " + str(j) + ": " + str(features[j]),
            fontsize="x-large",
        )
        plt.xlabel("Value", fontsize="x-large")
        plt.ylabel("Frequency", fontsize="x-large")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        output_filename = (
            file_path.removesuffix("_esindy.pkl")
            + "/E-SINDy/histogram"
            + str(i)
            + "feature"
            + str(j)
            + ".pdf"
        )
        plt.savefig(output_filename)
        plt.close()
