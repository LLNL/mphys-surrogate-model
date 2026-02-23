# mphys-surrogate-model
This repository contains python scripts for training latent-space machine learning representations of warm rain droplet coalescence and 
is the companion to a paper currently in review, titled "Data-Driven Reduced Order Modeling for Warm Rain Microphysics".
Superdroplet-enabled simulations of this warm rain formation process provide high information-density training data upon which various 
data-driven model structures are trained. All structures share in common a latent-space discovery based on an autoencoder (*1*); differences lie
in the varying representation of time-evolving dynamics within the latent space, which utilize one of three model structures: (1) SINDy (*2*);
(2) a neural-network derivative; (3) a finite-time step autoregressor.

## Getting started
You may clone this repository and run the included scripts and notebooks contained in `training_scripts` and/or notebooks in `analysis_notebooks` on most platforms, 
provided the proper python packages are available.

### Prerequisites
You will need to create a Python environment (such as `venv` or `conda`) with certain package dependencies, including PyTorch. 
A basic list of required packages can be found in [requirements.txt](requirements.txt). You may also attempt to replicate an exact
environment for MacOS and python version 3.11.9 contained in [requirements_os.txt](requirements_os.txt) with the command
```aiignore
pip install -r requirements_os.txt
```
though package versions may not be compatible with all operating systems.

A pyproject.toml file is also included for uv users. After 
[installing uv](https://docs.astral.sh/uv/getting-started/installation/),
you should be able to run `uv sync` in the repository folder, and it will create a virtual environment for you that you can activate.
If you don't have the right python version available, you may need to run `uv python install 3.11.9` before running `uv sync`.

### Installing & Running
Simply clone this repository from github and run the desired script locally; all paths are specified as relative, and no further
installation is necessary.
```aiignore
git clone <repo_link>
python training_scripts/<script_name>.py
```

## Contents

### Source
The key ingredients to build and train a microphysics ROM are included in three files:
- `data_utils.py` includes resources for creating pytorch dataloaders, datasets, and for postprocessing data including a simple ODE solver;
- `models.py` includes various NN model structures used for the autoencoder and latent space dynamics.

For analysis,
- `diagnostics.py` computes various diagnostic quantities of model performance;
- `plotting.py` includes functionality to visualize model predictions and accuracy. 

### Training Scripts
Contains scripts to perform end-to-end training of an autoencoder coupled to 3 latent space dynamics representations (`train_ae_*.py`). 
Also contains scripts to analyze and optimize model performance.

### UQ
Includes scripts, including slurm submission scripts, to compute confidence intervals on model predictions using conformal prediction.

### Results & Figures
These directories contain select archived trained model weights as well as diagnostic quantities and figures used for illustration in the
associated manuscript.

### Datasets and data generation
Training data from the ERF model with SDM microphysics are generated and postprocessed on LC; postprocessed datasets used directly in 
training pipelines are included as netcdf files in `data/erf_data`. 

An additional training dataset pipeline is included for reference in
`data/pysdm`, which uses jupyter notebooks to generate and postprocess box-model SDM simulations using `PySDM`.
Generating SDM data using this pipeline necessitates [PySDM](https://github.com/open-atmos/PySDM) as an additional dependency:
- `pysdm` (available via `pip`)
- `pysdm-examples` development version: https://github.com/open-atmos/PySDM with `dvdlnr` added as an additional output product

### Documentation
Sphinx autodocs are available in the `docs` folder. The docs can be built running the `build_all_docs.sh` script
located in the same folder.

## Authors
Core contributors to this codebase include:
 * Emily de Jong (LLNL) - [ekdejong](https://github.com/ekdejong)
 * Nipun Gunawardena (LLNL) - [madvoid](https://github.com/madvoid)
 * Jonas Katona (Yale U.)
Other co-authors who made this work possible include Hassan Beydoun, Peter Caldwell, and Debo Ghosh.

To cite this work, please use:

de Jong et al. DUMMY CITATION WITH LINK TO PREPRINT

## Acknowledgements
Several scientific principles and training strategies in this work are based on published work by:

(*1*) K. D. Lamb, M. van Lier-Walqui, S. Santos, and H. Morrison. Reduced-Order Modeling for Linearized Representations of Microphysical
Process rates. *JAMES* 2024, DOI [10.1029/2023MS003918](https://doi.org/10.1029%2F2023MS003918).

(*2*) Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Data-driven discovery of coordinates and governing equations. 
*PNAS* 2019, DOI [10.1073/pnas.1906995116](https://www.pnas.org/doi/10.1073/pnas.1906995116).

## Release and License
This codebase has been released under LLNL-CODE-2012627 and is distributed under the terms of the BSD-Commercial License. 
See [LICENSE](LICENSE) and [NOTICE](NOTICE).