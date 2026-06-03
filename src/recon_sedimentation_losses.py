"""
Loss functions for sedimentation surrogate models.
Provides loss computation for predicting sedimentation flux.
"""

import torch
from nwi import hhat_M_to_h

# Global loss functions
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Tuple of (batch_x, batch_dx, batch_M) - only batch_x is used
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_flux, batch_m = batch

    # Forward pass for reconstruction
    pred_x_recon = model.decoder(model.encoder(batch_x)) # batch_x & batch_flux have dimensions this time
    vt = batch_flux / (batch_x + params["tol"])  # Compute terminal velocity from flux and DSD

    # Calculate losses
    loss_kl = divergence(
        torch.log(pred_x_recon / batch_m + params["tol"]),
        torch.log(batch_x / batch_m + params["tol"]),
    )
    loss_l2 = criterion(pred_x_recon * vt, batch_flux)  # Vt-weighted reconstruction loss to emphasize bins with higher sedimentation flux

    # Weighted total loss
    loss = loss_kl + params["loss_weight_recon_vt"] * loss_l2

    loss_dict = {
        "total": loss,
        "kl": loss_kl,
        "l2_vt": loss_l2,
    }

    return loss, loss_dict

def compute_sedimentation_loss(model, batch, params, device):
    """
    Loss for sedimentation flux prediction using derivative-based dynamics (SINDy or NN dzdt).

    The model architecture:
    - encoder: x (dimensioned DSD) -> h (dimensioned latent variables)
    - dynamics: [h_hat, M] -> [h] -> flux_pred (predicted flux)
    - decoder: [h] -> [h_hat, M] -> x_recon (reconstructed normalized DSD)

    Args:
        model: ComposedModel with encoder, decoder, and dynamics
        batch: Tuple of (batch_x, batch_flux, batch_M)
               - batch_x: normalized DSD [batch, 1, n_bins]
               - batch_flux: target sedimentation flux [batch, 1, n_bins]
               - batch_M: total mass [batch, 1, 1]
        params: Parameters dict with loss weights and tolerance
        device: Compute device

    Returns:
        total_loss, loss_dict

    Notes:
        The latent variables h have dimensions of mass (they are linear projections
        of the normalized DSD). The dynamics module receives h and should
        predict flux in each bin.

        TODO: Fill in the flux prediction equation/loss calculation
    """
    batch_x, batch_flux, batch_m = batch
    batch_vt = batch_flux / (batch_x + params["tol"])  # Compute terminal velocity from flux and DSD

    # 1. Reconstruction loss (in dimensionless DSD space)
    pred_x_recon = model.decoder(model.encoder(batch_x)) # batch_x & batch_flux have dimensions this time
    loss_kl = divergence(
        torch.log(pred_x_recon / batch_m + params["tol"]),
        torch.log(batch_x / batch_m + params["tol"]),
    )

    # 2. Vt-weighted reconstruction loss (to emphasize bins with higher sedimentation flux)
    vt = batch_flux / (batch_x + params["tol"])  # Compute terminal velocity from flux and DSD
    loss_vt = criterion(pred_x_recon * vt, batch_flux)  # Vt-weighted reconstruction loss to emphasize bins with higher sedimentation flux

    # 3. Latent tendency in h space
    h = model.encoder(batch_x)
    pred_dh = model.dzdt(h)  # Predict flux from latent variables
    batch_dh = model.encoder(batch_flux)
    loss_flux_dh = criterion(pred_dh, batch_dh)  # Loss on latent flux prediction

    # 4. Latent tendency in x space
    hh = h.copy().detach().requires_grad_(True)
    _, pred_dx = torch.func.jvp(model.decoder, (hh,), (pred_dh,))
    loss_flux_dx = criterion(pred_dx, batch_flux)

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_kl
        + params["loss_weight_recon_vt"] * loss_vt
        + params["loss_weight_dx"] * loss_flux_dx
        + params["loss_weight_dz"] * loss_flux_dh
    )

    loss_dict = {
        "total": loss,
        "recon": loss_kl,
        "recon_vt": loss_vt,
        "flux_dh": loss_flux_dh,
        "flux_dx": loss_flux_dx,
    }

    return loss, loss_dict


def get_loss_function(dynamics_type):
    """
    Get the appropriate loss function for sedimentation models.

    Args:
        dynamics_type: "sindy" or "nn_dzdt" (only these are supported for sedimentation)

    Returns:
        Loss function callable
    """
    if dynamics_type in ["sindy", "nn_dzdt"]:
        return compute_sedimentation_loss
    elif dynamics_type == "none":
        return compute_autoencoder_loss
    else:
        raise ValueError(
            f"Sedimentation models only support 'sindy' or 'nn_dzdt' dynamics, got: {dynamics_type}"
        )
