Training Scripts
================

The ``training_scripts/`` directory contains scripts for training and evaluating coalescence surrogate models.

Main Training Script
--------------------

.. automodule:: training_scripts.train_coalescence_model
   :members:
   :undoc-members:
   :show-inheritance:

The unified training script supports multiple model architectures:

* **AE-SINDy**: Autoencoder with SINDy dynamics
* **AE-NNdzdt**: Autoencoder with neural network dynamics (dz/dt)
* **AE-AR**: Autoencoder with autoregressive dynamics
* **Pure Autoencoder**: NNWI-style models without explicit dynamics

Parameter Configuration
-----------------------

For complete documentation of all training parameters, see: :doc:`../TRAINING_PARAMS`

Key parameter groups:

* **Model Architecture**: ``encoder_type``, ``decoder_type``, ``dynamics_type``
* **Training Settings**: ``learning_rate``, ``batch_size``, ``num_epochs``
* **Loss Weights**: Controls for reconstruction vs dynamics loss balance
* **Data Settings**: ``data_src`` (box or erf), training/test split

Quick Start
-----------

1. Edit the ``params`` dict in ``train_coalescence_model.py``
2. Run the script:

.. code-block:: bash

   cd training_scripts
   python train_coalescence_model.py

3. Results are saved to ``../results/coalescence/``

Example Configurations
----------------------

**SINDy Model**::

    params = {
        "encoder_type": "ffnn",
        "decoder_type": "ffnn",
        "dynamics_type": "sindy",
        "latent_dim": 3,
        "poly_order": 2,
        ...
    }

**Neural Network Dynamics**::

    params = {
        "encoder_type": "nwi",
        "decoder_type": "nwi_simple",
        "dynamics_type": "nn_dzdt",
        "layer_size": (100, 100, 100),
        ...
    }

**Autoregressive Model**::

    params = {
        "encoder_type": "ffnn",
        "decoder_type": "ffnn",
        "dynamics_type": "autoregressive",
        "n_lag": 1,
        ...
    }

Related Scripts
---------------

* ``optuna_study.py`` - Hyperparameter optimization with Optuna
* ``eval_models_testdata.py`` - Evaluate trained models on test datasets
* ``random_seed_variation.py`` - Test model stability across random seeds
