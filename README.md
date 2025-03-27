# mphys-surrogate-model

## Description
This repository contains scripts and python notebooks for generating and processing high-fidelity superdroplet simulations of warm clouds, and for exploring machine learning approaches to representing these data.


## Getting started
You may clone this repository and run the included scripts contained in `training_scripts` and/or notebooks in `analysis_notebooks` locally or on LC, 
provided the proper python packages are available. Detailed instructions to create a `conda` environment on Lassen are provided at 
\url{https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=734049710}. You may attempt to create an environment based on the included
`requirements_lassen.txt`, or locally using `requirements_os.txt`, but no guarantees this will work perfectly. Note that `pysindy` is incompatible 
with the Lassen architecture and is therefore a dependency only in the os version, even though it is necessary for most analysis notebooks.

When using Lassen, the command
  `module unload cuda`
is required after loading your python environment and prior to running any script with a `pytorch` dependency.

## Contents

### Data and data generation
Sample training data files are included in the `data/` directory as netcdf files. Training data from the ERF model is run and postprocessed on LC.
Training data that is generated from pysdm can easily be reproduced in the `generate_*_data.*py*` files using the `PySDM` package.
For generating data using `generate_1d_data.py`
- `pysdm` (available via `pip`)
- `pysdm-examples` development version: \url{https://github.com/open-atmos/PySDM} with `dvdlnr` added as an additional output product

### Source
The key ingredients are included in three files:
- `data_utils.py` includes resources for creating pytorch dataloaders, datasets, and for postprocessing data including a simple ODE solver
- `models.py` includes various NN model structures
- `training.py` includes a few basic and re-usable training workflows

### Analysis notebooks
Should only be run locally or within an environment that supports `pysindy` as well as `pytorch` and other dependencies.

### Training scripts
Also included in this folder are sample submission scripts for GPU training on Lassen.