"""
Coalescence Model Training Framework

This package provides a unified framework for training autoencoder-based
coalescence surrogate models with various dynamics representations.

Main modules:
- model_factory: Create composed models with any encoder/decoder/dynamics combination
- training_utils: Unified training loop and setup utilities
- recon_coalescence_losses: Loss functions for different dynamics types
- recon_sedimentation_losses: Loss functions for sedimentation dynamics with NWI
- save_utils: Model saving, plotting, and output management
- data_utils: Dataset loading and preprocessing
- diagnostics: Model evaluation metrics
- plotting: Visualization utilities
- models: Model architectures (FFNN encoder/decoder, dynamics modules)
- nwi: Neural Weighted Integration encoder/decoder
"""

__version__ = "1.0.0"

# Export commonly used functions for convenience
from src import (
    constants,
    data_utils,
    diagnostics,
    model_factory,
    models,
    nwi,
    plotting,
    recon_coalescence_losses,
    recon_sedimentation_losses,
    save_utils,
    training_utils,
)
