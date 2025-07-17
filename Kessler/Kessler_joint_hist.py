# script to plot joint/2D histograms between pairs of for all coefficients from the result of eindy.py

import numpy as np
import matplotlib.pyplot as plt
import pickle

# import pickle file as inputed name
file_path = input('What is the .pkl file path?')

# load coefficient samples from E-SINDy
with open(file_path, 'rb') as f:
    bootstraps, features = pickle.load(f)
# transpose so we can loop through each coefficient
bootstraps = np.transpose(bootstraps, (1,2,0))

# use Gaussian KDE to estimate the probablity density and use this for plotting

from sklearn.neighbors import KernelDensity

# plot joint distribution for user-provided variables and coefficients
print('For reference, here are the feature names:', features)
i = int(input('First variable?')) # query for first variable
j_i = int(input('The coefficient for which term?'))
k = int(input('Second variable?')) # query for second variable
j_k = int(input('The coefficient for which term?'))

# get data
data_i = bootstraps[i, j_i] # get distribution for coefficient j_i for variable i
data_k = bootstraps[k, j_k] # get distribution for coefficient j_k for variable k
data = np.vstack((data_i,data_k)).T # combine distributions to now train KDE

# get custom bandwidth
# Compute standard deviation for each dimension (or overall if using a scalar measure)
std_devs = np.std(data, axis=0)

# Scott's Rule for multivariate data: bandwidth factor = n^{-1/(d+4)}
scott_factor = bootstraps.shape[2] ** (-1 / (bootstraps.shape[0] + 4))
# average bandwidth across each dimension
scott_bandwidth = np.mean(std_devs)* scott_factor

adjustment_factor = 0.5  # decrease value for more jagged density, increase for smoother density
kde = KernelDensity(kernel='epanechnikov', bandwidth=scott_bandwidth * adjustment_factor) # define KDE
kde.fit(data) # fit KDE

# get bounds for grid
x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
# create a higher resolution grid
grid_resolution = 200
X, Y = np.mgrid[x_min:x_max:grid_resolution*1j, y_min:y_max:grid_resolution*1j]
grid_coords = np.vstack([X.ravel(), Y.ravel()]).T

# evaluate kernel density on the grid
# score_samples method the log-density; exponentiate to get density values
Z = np.exp(kde.score_samples(grid_coords))
Z = Z.reshape(X.shape)

# Determine contour levels: using np.linspace for even spacing
num_levels = 50  # Adjust this value as needed
levels = np.linspace(Z.min(), Z.max(), num_levels)

from matplotlib.colors import LogNorm

# plot the results
plt.figure(figsize=(8, 6))
# Use contourf to fill contours of the density function
plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.scatter(data[:, 0], data[:, 1], s=5, facecolor='white', edgecolor='black', alpha=0.5)
plt.title('2D Probability Density from KDE')
plt.xlabel(str(features[j_i])+' for variable '+str(i))
plt.ylabel(str(features[j_k])+' for variable '+str(k))
plt.tight_layout()

# Save the plot to a file
output_filename = file_path.removesuffix("_esindy.pkl")+"/E-SINDy/2Dhistogram"+str(i)+"feature"+str(j_i)+'and'+str(k)+"feature"+str(j_k)+".pdf"
plt.savefig(output_filename)
plt.savefig("Kessler_joint_hist.pdf")
plt.close()
