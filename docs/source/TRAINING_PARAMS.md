# Training Parameters Guide

This document describes all available parameters for training coalescence surrogate models using `training_scripts/coalescence/train_coalescence_model.py`.

## Model Architecture Parameters

| Parameter | Type | Required | Options/Range | Description |
|-----------|------|----------|---------------|-------------|
| `encoder_type` | string | Yes | `"ffnn"`, `"nwi"` | Type of encoder. `"ffnn"` = feed-forward neural network; `"nwi"` = Normalizing Weight Interpolation (more interpretable) |
| `decoder_type` | string | Yes | `"ffnn"`, `"nwi_simple"`, `"nwi_deep"` | Type of decoder. `"ffnn"` = feed-forward; `"nwi_simple"` = simple NWI; `"nwi_deep"` = deep NWI with more layers |
| `dynamics_type` | string | Yes | `"sindy"`, `"nn_dzdt"`, `"autoregressive"`, `"none"` | Type of dynamics model. `"sindy"` = Sparse Identification of Nonlinear Dynamics; `"nn_dzdt"` = NN for learning dz/dt; `"autoregressive"` = direct next-state prediction; `"none"` = pure autoencoder |
| `latent_dim` | int | Yes | 2-10 (typical) | Dimension of latent space. Lower values = more compression but may lose information |

## Architecture-Specific Parameters

### NWI-Specific (when `encoder_type="nwi"` or `decoder_type` contains `"nwi"`)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_blocks` | int | 3 | 2-5 | Number of residual blocks in NWI decoder |
| `hidden_size` | int | 128 | 64-256 | Hidden layer size in NWI modules |

### SINDy-Specific (when `dynamics_type="sindy"`)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `poly_order` | int | 2 | 1 (linear), 2 (quadratic), 3 (cubic), etc. | Polynomial order for SINDy library. Higher orders = more complex dynamics but risk overfitting |

### NN dzdt-Specific (when `dynamics_type="nn_dzdt"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer_size` | tuple of ints | `(100, 100, 100)` | Hidden layer sizes for neural network dynamics. Example: `(50, 50)` for 2 hidden layers of size 50 |

### Autoregressive-Specific (when `dynamics_type="autoregressive"`)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_lag` | int | 1 | 1-5 | Number of past time steps to use for prediction. Higher lag captures more temporal dependencies but increases complexity |

## Data Parameters

| Parameter | Type | Required | Options | Description |
|-----------|------|----------|---------|-------------|
| `data_src` | string | Yes | `"box"`, `"erf"` | Source of training data. `"box"` = box model; `"erf"` = ERF model |

## Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `random_seed` | int | 10 | - | Random seed for reproducibility. Set to same value for reproducible results |
| `num_epochs` | int | 100 | 50-500 | Number of training epochs |
| `batch_size` | int | 25 | - | Training batch size. Larger batches = more stable gradients but more memory |
| `learning_rate` | float | 0.004 | 0.0001-0.01 | Initial learning rate for optimizer. Can be tuned with hyperparameter optimization |
| `wd` | float | 1e-3 | 0-0.01 | Weight decay (L2 regularization) |
| `lr_sched` | bool | `True` | - | Whether to use learning rate scheduler (ReduceLROnPlateau) |
| `patience` | int | 50 | - | Early stopping patience (epochs without improvement). Set to `None` to disable |
| `print_frequency` | int | 1 | - | How often (in epochs) to print training progress. Set higher to reduce console output |

## Loss Parameters

### Common Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tol` | float | 1e-8 | Tolerance added to avoid log(0) in KL divergence. Should be small positive number |

### SINDy & NN dzdt Loss Weights

For `dynamics_type="sindy"` or `"nn_dzdt"`, the loss is:
```
loss = loss_weight_recon * L_recon + loss_weight_dx * L_dx + loss_weight_dz * L_dz
```

| Parameter | Type | Default | Typical Values | Description |
|-----------|------|---------|----------------|-------------|
| `loss_weight_recon` | float | Auto-computed (Champion et al.) | ~1.0 | Weight for reconstruction loss. Set manually to override |
| `loss_weight_dx` | float | Auto-computed (Champion et al.) | ~1000-100000 | Weight for dx/dt consistency loss. Set manually to override |
| `loss_weight_dz` | float | Auto-computed (Champion et al.) | ~10-1000 | Weight for dz/dt consistency loss. Set manually to override |
| `lambda1_metaweight` | float | 0.5 | 0-1 | Meta-parameter for Champion et al. weight calculation. Only used if loss weights not manually specified. Lower values reduce dx/dt loss influence |

### Autoregressive Loss Weights

For `dynamics_type="autoregressive"`, the loss is:
```
loss = w_recon * L_recon + w_dx * L_dx + w_dz * L_dz
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_recon` | float | 1.0 | Weight for reconstruction loss at t=0 |
| `w_dx` | float | 1.0 | Weight for prediction KL divergence |
| `w_dz` | float | 0.1 | Weight for latent space prediction loss |

### Pure Autoencoder Loss Weights

For `dynamics_type="none"`, the loss is:
```
loss = L_kl + loss_weight_l2 * L_l2
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_weight_l2` | float | 0.01 | Weight for L2 reconstruction loss |

## Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save` | bool | `True` | Whether to save model artifacts and plots |
| `show_plots` | bool | `False` | Whether to display plots interactively. Set to `True` for interactive visualization |

## Example Configurations

### AE-SINDy with FFNN
```python
params = {
    "encoder_type": "ffnn",
    "decoder_type": "ffnn",
    "dynamics_type": "sindy",
    "latent_dim": 3,
    "poly_order": 2,
    "data_src": "erf",
    "num_epochs": 100,
    "batch_size": 25,
    "learning_rate": 0.004,
}
```

### AE-SINDy with NWI
```python
params = {
    "encoder_type": "nwi",
    "decoder_type": "nwi_simple",
    "dynamics_type": "sindy",
    "num_blocks": 3,
    "hidden_size": 128,
    "latent_dim": 3,
    "poly_order": 2,
    "data_src": "erf",
    "num_epochs": 100,
}
```

### AE-NNdzdt
```python
params = {
    "encoder_type": "nwi",
    "decoder_type": "nwi_simple",
    "dynamics_type": "nn_dzdt",
    "num_blocks": 3,
    "hidden_size": 128,
    "layer_size": (100, 100, 100),
    "latent_dim": 3,
    "data_src": "erf",
}
```

### AE-Autoregressive
```python
params = {
    "encoder_type": "ffnn",
    "decoder_type": "ffnn",
    "dynamics_type": "autoregressive",
    "n_lag": 1,
    "latent_dim": 3,
    "data_src": "erf",
    "w_dx": 1.0,
    "w_recon": 1.0,
    "w_dz": 0.1,
}
```

### Pure NNWI Autoencoder
```python
params = {
    "encoder_type": "nwi",
    "decoder_type": "nwi_simple",
    "dynamics_type": "none",
    "num_blocks": 3,
    "hidden_size": 128,
    "latent_dim": 3,
    "data_src": "erf",
    "loss_weight_l2": 0.01,
}
```

## Output Directory Structure

Models are saved to:
```
trained_models/{encoder_type}_{decoder_type}_{dynamics_type}/{case_name}/
```

Where `case_name` includes:
- Timestamp
- Data source
- Latent dimension
- Key hyperparameters
- Loss weights

Example:
```
trained_models/ffnn_ffnn_sindy/20260604_130740_erf_latent3_order2_ep1_lr4e-03_bs25_w1.0-405.8-40583/
```

Each directory contains:
- `model.pth` - Model weights
- `params.json` - Full parameters used
- `losses.pkl` - Training/test losses
- Various plots (`losses.png`, `reconstructions.png`, `predictions.png`, `trajectories.png`, etc.)

## Running Training Scripts

All training scripts should be run from the repository root:

```bash
# Coalescence model
python training_scripts/coalescence/train_coalescence_model.py

# Sedimentation model
python training_scripts/sedimentation/train_sedimentation_model.py
```

## Testing

Verify your installation and model interfaces work correctly:

```bash
# Quick interface tests (10 configurations, ~30 seconds)
python tests/test_unified_interface.py

# Full end-to-end training tests (8 configurations, ~5 minutes)
python tests/test_all_training_configs.py
```
