# mphys-surrogate-model

This repository contains python scripts for training latent-space machine learning representations of warm rain droplet coalescence (*1*) with uncertainty quantification (*2*) and 
is the companion to a paper titled "Data-Driven Reduced Order Modeling for Warm Rain Microphysics".
Superdroplet-enabled simulations of this warm rain formation process provide high information-density training data upon which various 
data-driven model structures are trained. All structures share in common a latent-space discovery based on an autoencoder (*3*); differences lie
in the varying representation of time-evolving dynamics within the latent space, which utilize one of three model structures: (1) SINDy (*4*);
(2) a neural-network derivative; (3) a finite-time step autoregressor.

## Getting Started

### Prerequisites

You will need to create a Python environment (such as `venv` or `conda`) with certain package dependencies, including PyTorch. 

**Option 1: Using uv (recommended)**

After [installing uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```
This will create a virtual environment with all dependencies. If you don't have Python 3.11.9 available:
```bash
uv python install 3.11.9
uv sync
```

**Option 2: Using pip**

A basic list of required packages can be found in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```

For a complete environment specification (MacOS, Python 3.11.9):
```bash
pip install -r requirements_os.txt
```
Note: package versions may not be compatible with all operating systems.

### Quick Start

1. Clone the repository:
```bash
git clone <repo_link>
cd mphys-surrogate-model
```

2. Set up your environment (see Prerequisites above)

3. Train a model:
```bash
python training_scripts/coalescence/train_coalescence_model.py
```

4. Run tests to verify installation:
```bash
python tests/test_unified_interface.py
```

5. View the documentation (see [Building Documentation](#building-documentation) below)

## Repository Structure

### `src/`
Core library modules for building and training microphysics reduced-order models:

- **`data_utils.py`** - PyTorch dataloaders, datasets, and postprocessing utilities including ODE solvers
- **`models.py`** - Neural network architectures for autoencoders and latent space dynamics
- **`model_factory.py`** - Factory functions for creating model instances from configuration parameters
- **`nwi.py`** - Neural Weighted Integral (NWI) encoder/decoder implementations
- **`recon_coalescence_losses.py`** - Loss functions for reconstruction and coalescence physics
- **`training_utils.py`** - Training loop utilities and helper functions
- **`save_utils.py`** - Model checkpointing and result serialization
- **`diagnostics.py`** - Model performance diagnostics and metrics
- **`plotting.py`** - Visualization tools for predictions and accuracy analysis

### `training_scripts/`
End-to-end training and evaluation scripts organized by model type:

#### `training_scripts/coalescence/`
Scripts for coalescence microphysics models:
- **`train_coalescence_model.py`** - Unified training script supporting multiple architectures (AE-SINDy, AE-NNdzdt, AE-AR)
- **`optuna_study.py`** - Hyperparameter optimization using Optuna
- **`eval_models_testdata.py`** - Evaluate trained models on test datasets
- **`random_seed_variation.py`** - Test model stability across random initializations
- **`export_to_ftorch.py`** - Export trained models to FTorch format for Fortran integration

#### `training_scripts/sedimentation/`
Scripts for sedimentation microphysics models:
- **`train_sedimentation_model.py`** - Training script for sedimentation flux prediction models

See [docs/TRAINING_PARAMS.md](docs/TRAINING_PARAMS.md) for complete parameter documentation.

### `tests/`
Test suite for validating model interfaces and training pipelines:
- **`test_unified_interface.py`** - Unit tests for encoder/decoder interface with all combinations
- **`test_all_training_configs.py`** - End-to-end training pipeline tests

### `uq/`
Uncertainty quantification scripts using conformal prediction, including SLURM submission scripts for HPC environments.

### `results/`
Archived trained model weights, diagnostic outputs, and analysis notebooks. Organized by model type (NNWI, Optuna studies, etc.).

### `figures/`
Figures and visualizations for the associated manuscript.

### `data/`
Training datasets and data generation pipelines:

- **`erf_data/`** - Postprocessed netCDF datasets from ERF model with SDM microphysics
- **`pysdm/`** - Jupyter notebooks for generating box-model SDM simulations using [PySDM](https://github.com/open-atmos/PySDM)

Note: PySDM data generation requires additional dependencies (see PySDM documentation).

## Building Documentation

The repository uses Sphinx for API documentation. To build the docs:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view.

For a complete rebuild:
```bash
cd docs
./build_all_docs.sh
```

To see all available build formats:
```bash
cd docs
make help
```

## Authors
Core contributors to this codebase include:
 * Emily de Jong (LLNL) - [ekdejong](https://github.com/ekdejong)
 * Nipun Gunawardena (LLNL) - [madvoid](https://github.com/madvoid)
 * Jonas Katona (Yale U.)
Other co-authors who made this work possible include Hassan Beydoun, Peter Caldwell, and Debo Ghosh.

## Citations

To cite this work, please cite:

(*1*) E. K. de Jong, N. Gunawardena, J. E. Katona, et al. Data-Driven Reduced Order Modeling for Warm Rain Microphysics. *JGR:ML&C* 2026, DOI [10.22541/au.176220214.46858428/v1] (https://www.authorea.com/doi/full/10.22541/au.176220214.46858428/v1) (in press).

(*2*) J. E. Katona, E. K. de Jong, N. Gunawardena. Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics. *Accepted to 2025 NeurIPS workshop on Machine Learning for Physical Sciences*, (https://arxiv.org/abs/2511.04534).

## Acknowledgements
Several scientific principles and training strategies in this work are based on published work by:

(*3*) K. D. Lamb, M. van Lier-Walqui, S. Santos, and H. Morrison. Reduced-Order Modeling for Linearized Representations of Microphysical
Process rates. *JAMES* 2024, DOI [10.1029/2023MS003918](https://doi.org/10.1029%2F2023MS003918).

(*4*) Kathleen Champion, Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Data-driven discovery of coordinates and governing equations. 
*PNAS* 2019, DOI [10.1073/pnas.1906995116](https://www.pnas.org/doi/10.1073/pnas.1906995116).

## Release and License
This codebase has been released under LLNL-CODE-2012627 and is distributed under the terms of the BSD-Commercial License. 
See [LICENSE](LICENSE) and [NOTICE](NOTICE).