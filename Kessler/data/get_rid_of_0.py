from netCDF4 import Dataset

# Open the NetCDF file in append mode
with Dataset('rico_test.nc', 'r+') as nc:
    # Access the variable 't'
    t_var = nc.variables['t']

    # Overwrite the first element
    t_var[0] = 72000