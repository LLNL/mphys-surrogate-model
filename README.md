# mphys-surrogate-model

## Description
This repository contains scripts and python notebooks for generating and processing high-fidelity superdroplet simulations of warm clouds, and for exploring machine learning approaches to representing these data.


## Getting started
You may clone this repository and run the included scripts and notebooks locally or on LC, provided the proper python packages are available. Detailed instructions to create a `conda` environment on Lassen are provided at \url{https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=734049710}.

When using Lassen, the command
  `module unload cuda`
is required after loading your python environment and prior to running any script with a `pytorch` dependency.

## Contents & Dependencies
Running the scripts and notebooks in this repository requires the following python dependencies:

For generating data using `generate_1d_data.py`
- `pysdm` (available via `pip`)
- `pysdm-examples` development version: \url{https://github.com/open-atmos/PySDM} with `dvdlnr` added as an additional output product

For processing and visualizing data, including PCA, in `preprocess_pysdm.ipynb` 
- `numpy`
- `xarray`
- `dask`
- `scipy`
- `sklearn`
- `matplotlib`
available via `conda` or `pip`

The ML model setup and training in `pysdm_pytorch.ipynb` or in `train_vae.py` with module `models.py` has additional dependence on 
- `pytorch`

## Future plans:
[] Test equation-learning capability with either `SINDY` or `LaSDI`
[] Use output data from ERF and/or a more realistic 3D simulation