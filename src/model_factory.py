"""
Model factory for composable encoder/decoder/dynamics architecture.
Allows mixing and matching different encoder types, decoder types, and dynamics models.
"""

import torch
from torch import nn

from src import models, nwi


class ComposedModel(nn.Module):
    """
    Generic composed model with encoder, decoder, and optional dynamics.
    Provides consistent interface regardless of component choices.
    """

    def __init__(self, encoder, decoder, dynamics=None, dynamics_type="none"):
        super(ComposedModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dzdt = dynamics  # Can be None for pure autoencoders
        self.dynamics_type = dynamics_type

    def forward(self, x, *args):
        """
        Forward pass depends on dynamics type:
        - autoregressive: forward(x, M) -> reconstructed x at t+1
        - sindy/nn_dzdt: forward(x, M) -> (dz/dt, dM/dt)
        - none: forward(x) -> reconstructed x
        """
        if self.dynamics_type == "autoregressive":
            # x is lagged inputs, args[0] is M
            M = args[0]
            latent0 = []
            n_lag = getattr(self.dzdt, "n_lag", 1)
            for t in range(n_lag):
                latent0.append(self.encoder(x[:, t, :]).unsqueeze(1))
            latent0 = torch.cat(latent0, dim=2)
            latent0_M = torch.cat([latent0, M], dim=2)
            latent1_M = self.dzdt(latent0_M)
            latent1 = latent1_M[:, :, :-1]
            dM = latent1_M[:, :, -1]  # Extract dM prediction
            bin1 = self.decoder(latent1)
            return bin1, dM

        elif self.dynamics_type in ["sindy", "nn_dzdt"]:
            # x is current state, args[0] is M
            M = args[0]
            z0 = self.encoder(x)
            dzMdt = self.dzdt(z0, M)
            dzdt = dzMdt[:, :, :-1]  # Latent dynamics
            dMdt = dzMdt[:, :, -1]    # Mass derivative (should be ~0 for coalescence)
            return dzdt, dMdt

        elif self.dynamics_type == "none":
            # Pure autoencoder
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon

        else:
            raise ValueError(f"Unknown dynamics_type: {self.dynamics_type}")


# Registry of available components
ENCODER_REGISTRY = {
    "ffnn": models.FFNNEncoder,
    "nwi": nwi.LinearEncoder,
    # "cnn": None,  # Placeholder for future CNN encoder
}

DECODER_REGISTRY = {
    "ffnn": models.FFNNDecoder,
    "nwi_simple": nwi.SimpleDecoder,
    "nwi_deep": nwi.DeepDecoder,
}

DYNAMICS_REGISTRY = {
    "sindy": models.SINDyDeriv,
    "nn_dzdt": models.NNDerivatives,
    "autoregressive": models.Autoregressive,
    "none": None,
}


def create_encoder(encoder_type, n_bins, n_latent, params):
    """
    Create encoder based on type.

    Args:
        encoder_type: "ffnn", "nwi", or "cnn"
        n_bins: Number of distribution bins
        n_latent: Latent dimension
        params: Additional parameters (unused for most encoders)

    Returns:
        Encoder module
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Available: {list(ENCODER_REGISTRY.keys())}")

    encoder_cls = ENCODER_REGISTRY[encoder_type]

    if encoder_type == "ffnn":
        return encoder_cls(n_bins=n_bins, n_latent=n_latent)
    elif encoder_type == "nwi":
        return encoder_cls(n_bins=n_bins, n_latent=n_latent)
    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not yet implemented")


def create_decoder(decoder_type, n_bins, n_latent, params):
    """
    Create decoder based on type.

    Args:
        decoder_type: "ffnn", "nwi_simple", or "nwi_deep"
        n_bins: Number of distribution bins
        n_latent: Latent dimension
        params: Additional parameters (hidden_size, num_blocks for NWI)

    Returns:
        Decoder module
    """
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Available: {list(DECODER_REGISTRY.keys())}")

    decoder_cls = DECODER_REGISTRY[decoder_type]

    if decoder_type == "ffnn":
        return decoder_cls(n_bins=n_bins, n_latent=n_latent, distribution=True)
    elif decoder_type in ["nwi_simple", "nwi_deep"]:
        hidden_size = params.get("hidden_size", 128)
        num_blocks = params.get("num_blocks", 3)
        return decoder_cls(
            n_bins=n_bins,
            n_latent=n_latent,
            hidden_features=hidden_size,
            num_blocks=num_blocks,
        )
    else:
        raise NotImplementedError(f"Decoder type {decoder_type} not yet implemented")


def create_dynamics(dynamics_type, n_latent, params):
    """
    Create dynamics model based on type.

    Args:
        dynamics_type: "sindy", "nn_dzdt", "autoregressive", or "none"
        n_latent: Latent dimension
        params: Additional parameters (poly_order for SINDy, layer_size for NN, n_lag for AR)

    Returns:
        Dynamics module or None
    """
    if dynamics_type not in DYNAMICS_REGISTRY:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}. Available: {list(DYNAMICS_REGISTRY.keys())}")

    if dynamics_type == "none":
        return None

    dynamics_cls = DYNAMICS_REGISTRY[dynamics_type]

    if dynamics_type == "sindy":
        poly_order = params.get("poly_order", 2)
        return dynamics_cls(
            n_latent=n_latent + 1,  # +1 for mass coordinate
            poly_order=poly_order,
            use_thresholds=False,
        )
    elif dynamics_type == "nn_dzdt":
        layer_size = params.get("layer_size", (100, 100, 100))
        return dynamics_cls(n_latent=n_latent + 1, layer_size=layer_size)
    elif dynamics_type == "autoregressive":
        n_lag = params.get("n_lag", 1)
        layer_size = params.get("layer_size", (100, 100, 100))
        return dynamics_cls(
            n_bins=n_latent + 1,
            n_bins_in=n_latent * n_lag + 1,
            layer_size=layer_size,
        )
    else:
        raise NotImplementedError(f"Dynamics type {dynamics_type} not yet implemented")


def create_model(encoder_type, decoder_type, dynamics_type, params, n_bins):
    """
    Factory function to create a composed model from components.

    Args:
        encoder_type: "ffnn", "nwi", or "cnn"
        decoder_type: "ffnn", "nwi_simple", or "nwi_deep"
        dynamics_type: "sindy", "nn_dzdt", "autoregressive", or "none"
        params: Dict with model parameters (latent_dim, poly_order, layer_size, etc.)
        n_bins: Number of distribution bins

    Returns:
        ComposedModel with encoder, decoder, and dynamics components

    Notes:
        - For dynamics_type in ["sindy", "nn_dzdt"], forward() returns (dzdt, dMdt)
          where dMdt should be ~0 for mass-conserving processes like coalescence
        - For dynamics_type "autoregressive", forward() returns (bin1, dM)
        - For dynamics_type "none", forward() returns x_recon

    Example:
        # AE-SINDy with FFNN encoder
        model = create_model("ffnn", "ffnn", "sindy", params, 64)

        # AE-SINDy with NWI encoder
        model = create_model("nwi", "nwi_simple", "sindy", params, 64)

        # Pure NNWI autoencoder
        model = create_model("nwi", "nwi_simple", "none", params, 64)

        # AE-AR with FFNN
        model = create_model("ffnn", "ffnn", "autoregressive", params, 64)
    """
    n_latent = params["latent_dim"]

    # Create components
    encoder = create_encoder(encoder_type, n_bins, n_latent, params)
    decoder = create_decoder(decoder_type, n_bins, n_latent, params)
    dynamics = create_dynamics(dynamics_type, n_latent, params)

    # Compose into single model
    model = ComposedModel(encoder, decoder, dynamics, dynamics_type)

    return model
