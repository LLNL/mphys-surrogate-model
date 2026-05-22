# Training Parameters Guide

This document describes all available parameters for training coalescence surrogate models using `train_coalescence_model.py`.

## Model Architecture Parameters

### `encoder_type` (string, required)
Type of encoder to use.
- **Options**: `"ffnn"`, `"nwi"`
- `"ffnn"`: Feed-forward neural network encoder
- `"nwi"`: Normalizing Weight Interpolation encoder (more interpretable)

### `decoder_type` (string, required)
Type of decoder to use.
- **Options**: `"ffnn"`, `"nwi_simple"`, `"nwi_deep"`
- `"ffnn"`: Feed-forward neural network decoder
- `"nwi_simple"`: Simple NWI decoder
- `"nwi_deep"`: Deep NWI decoder with more layers

### `dynamics_type` (string, required)
Type of dynamics model.
- **Options**: `"sindy"`, `"nn_dzdt"`, `"autoregressive"`, `"none"`
- `"sindy"`: Sparse Identification of Nonlinear Dynamics (polynomial dynamics)
- `"nn_dzdt"`: Neural network for learning dz/dt
- `"autoregressive"`: Direct prediction of next state from lagged inputs
- `"none"`: Pure autoencoder with no dynamics (e.g., for NNWI)

### `latent_dim` (int, required)
Dimension of the latent space.
- **Typical values**: 2-10
- Lower values give more compression but may lose information

## Architecture-Specific Parameters

### NWI-Specific (when `encoder_type="nwi"` or `decoder_type` contains `"nwi"`)

#### `num_blocks` (int)
Number of residual blocks in NWI decoder.
- **Default**: 3
- **Typical range**: 2-5

#### `hidden_size` (int)
Hidden layer size in NWI modules.
- **Default**: 128
- **Typical range**: 64-256

### SINDy-Specific (when `dynamics_type="sindy"`)

#### `poly_order` (int)
Polynomial order for SINDy library.
- **Default**: 2
- **Options**: 1 (linear), 2 (quadratic), 3 (cubic), etc.
- Higher orders allow more complex dynamics but risk overfitting

### NN dzdt-Specific (when `dynamics_type="nn_dzdt"`)

#### `layer_size` (tuple of ints)
Hidden layer sizes for neural network dynamics.
- **Default**: `(100, 100, 100)`
- **Example**: `(50, 50)` for 2 hidden layers of size 50

### Autoregressive-Specific (when `dynamics_type="autoregressive"`)

#### `n_lag` (int)
Number of past time steps to use for prediction.
- **Default**: 1
- **Typical range**: 1-5
- Higher lag may capture more temporal dependencies but increases complexity

## Data Parameters

### `data_src` (string, required)
Source of training data.
- **Options**: `"box"`, `"erf"`
- `"box"`: Box model data
- `"erf"`: ERF model data

## Training Parameters

### `random_seed` (int)
Random seed for reproducibility.
- **Default**: 10
- Set to same value for reproducible results

### `num_epochs` (int)
Number of training epochs.
- **Default**: 100
- **Typical range**: 50-500

### `batch_size` (int)
Training batch size.
- **Default**: 25
- Larger batches = more stable gradients but more memory

### `learning_rate` (float)
Initial learning rate for optimizer.
- **Default**: 0.004
- **Typical range**: 0.0001-0.01
- Can be tuned with hyperparameter optimization

### `wd` (float)
Weight decay (L2 regularization).
- **Default**: 1e-3
- **Typical range**: 0-0.01

### `lr_sched` (bool)
Whether to use learning rate scheduler.
- **Default**: `True`
- Uses ReduceLROnPlateau scheduler

### `patience` (int)
Early stopping patience (epochs without improvement).
- **Default**: 50
- Set to `None` or omit to disable early stopping

### `print_frequency` (int)
How often (in epochs) to print training progress.
- **Default**: 1
- Set higher to reduce console output

## Loss Parameters

### Common Loss Parameters

#### `tol` (float)
Tolerance added to avoid log(0) in KL divergence.
- **Default**: 1e-8
- Should be small positive number

### SINDy & NN dzdt Loss Weights

For `dynamics_type="sindy"` or `"nn_dzdt"`, the loss is:
```
loss = loss_weight_recon * L_recon + loss_weight_dx * L_dx + loss_weight_dz * L_dz
```

#### `loss_weight_recon` (float, optional)
Weight for reconstruction loss.
- **Default**: Computed via Champion et al. method (typically ~1.0)
- Set manually to override automatic computation

#### `loss_weight_dx` (float, optional)
Weight for dx/dt consistency loss.
- **Default**: Computed via Champion et al. method (typically ~1000-100000)
- Set manually to override automatic computation

#### `loss_weight_dz` (float, optional)
Weight for dz/dt consistency loss.
- **Default**: Computed via Champion et al. method (typically ~10-1000)
- Set manually to override automatic computation

#### `lambda1_metaweight` (float)
Meta-parameter for Champion et al. weight calculation.
- **Default**: 0.5
- **Range**: 0-1
- Only used if loss weights not manually specified
- Lower values reduce the influence of dx/dt loss

### Autoregressive Loss Weights

For `dynamics_type="autoregressive"`, the loss is:
```
loss = w_recon * L_recon + w_dx * L_dx + w_dz * L_dz
```

#### `w_recon` (float)
Weight for reconstruction loss at t=0.
- **Default**: 1.0

#### `w_dx` (float)
Weight for prediction KL divergence.
- **Default**: 1.0

#### `w_dz` (float)
Weight for latent space prediction loss.
- **Default**: 0.1

### Pure Autoencoder Loss Weights

For `dynamics_type="none"`, the loss is:
```
loss = L_kl + loss_weight_l2 * L_l2
```

#### `loss_weight_l2` (float)
Weight for L2 reconstruction loss.
- **Default**: 0.01

## Output Parameters

### `save` (bool)
Whether to save model artifacts and plots.
- **Default**: `True`

### `show_plots` (bool)
Whether to display plots interactively.
- **Default**: `False`
- Set to `True` for interactive visualization

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
trained_models/ae_{encoder_type}_{dynamics_type}/{case_name}/
```

Where `case_name` includes:
- Encoder, decoder, and dynamics types
- Key hyperparameters
- Timestamp for uniqueness

Example:
```
trained_models/ae_nwi_sindy/nwi_nwi_simple_sindy_erf_latent3_order2_tr100_lr0.004_bs25_weights1.0-561.06-56106.47_20260521_143022/
```

Each directory contains:
- `{case_name}.pth` - Model weights
- `{case_name}_params.json` - Full parameters used
- `{case_name}.pkl` - Training/test losses
- Various plots (`_losses.png`, `_reconstructions.png`, `_predictions.png`, etc.)
