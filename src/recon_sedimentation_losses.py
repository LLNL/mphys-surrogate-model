"""
Loss functions for sedimentation surrogate models.
Provides loss computation for predicting sedimentation flux.
"""

import torch
import torch.nn.functional as F

# Global loss functions
mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Tuple of (batch_x, batch_flux, batch_m)
               - batch_x: dimensioned DSD
               - batch_flux: sedimentation flux
               - batch_m: total mass (redundant, encoder extracts it)
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_flux, batch_m = batch

    # Forward pass for reconstruction
    # Encoder outputs Z = [z, M], decoder takes Z and outputs dimensioned DSD
    pred_x_recon = model.decoder(model.encoder(batch_x))
    vt = batch_flux / (batch_x + params["tol"])  # Compute terminal velocity from flux and DSD

    # Calculate losses in normalized space for KL divergence
    loss_kl = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )
    # Vt-weighted reconstruction loss to emphasize bins with higher sedimentation flux
    loss_l2 = mse(pred_x_recon * vt, batch_flux)

    # Weighted total loss
    loss = loss_kl + params["loss_weight_recon_vt"] * loss_l2

    loss_dict = {
        "total": loss,
        "kl": loss_kl,
        "l2_vt": params["loss_weight_recon_vt"] * loss_l2,
    }

    return loss, loss_dict

def compute_sedimentation_loss(model, batch, params, device):
    """
    Loss for sedimentation flux prediction using derivative-based dynamics (SINDy or NN dzdt).

    The model architecture:
    - encoder: x (dimensioned DSD) -> Z = [z, M] (dimensioned latent variables)
    - dynamics: Z -> flux_pred (predicted flux in latent space)
    - decoder: Z -> x_recon (reconstructed dimensioned DSD)

    Args:
        model: ComposedModel with encoder, decoder, and dynamics
        batch: Tuple of (batch_x, batch_flux, batch_M)
               - batch_x: dimensioned DSD [batch, 1, n_bins]
               - batch_flux: target sedimentation flux [batch, 1, n_bins]
               - batch_M: total mass [batch, 1, 1] (redundant, encoder extracts it)
        params: Parameters dict with loss weights and tolerance
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_flux, batch_m = batch

    # 1. Reconstruction loss (in normalized DSD space)
    Z = model.encoder(batch_x)
    pred_x_recon = model.decoder(Z)
    loss_kl = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    # 2. Vt-weighted reconstruction loss (to emphasize bins with higher sedimentation flux)
    vt = batch_flux / (batch_x + params["tol"])  # Compute terminal velocity from flux and DSD
    loss_vt = mse(pred_x_recon * vt, batch_flux)

    # 3. Latent flux prediction in Z space: log-space MSE
    pred_dZ = model.dzdt(Z)  # Predict flux in latent space
    batch_dZ = model.encoder(batch_flux)
    loss_flux_dh = mse(torch.log(pred_dZ + params['tol']), torch.log(batch_dZ + params['tol']))

    # 4. Flux prediction in x space using JVP
    Z_detached = Z.clone().detach().requires_grad_(True)
    _, pred_flux = torch.func.jvp(model.decoder, (Z_detached,), (pred_dZ,))
    loss_flux_dx = mse(pred_flux, batch_flux)

    # 5. Penalty for a negative flux prediction in x space
    loss_negFlux = mae(F.relu(-pred_flux), F.relu(-batch_flux))

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_kl
        + params["loss_weight_recon_vt"] * loss_vt
        + params["loss_weight_dx"] * loss_flux_dx
        + params["loss_weight_dz"] * loss_flux_dh
        + params["loss_weight_negFlux"] * loss_negFlux
    )

    loss_dict = {
        "total": loss,
        "recon": params["loss_weight_recon"] * loss_kl,
        "recon_vt":  params["loss_weight_recon_vt"] * loss_vt,
        "flux_dh": params["loss_weight_dz"] * loss_flux_dh,
        "flux_dx": params["loss_weight_dx"] * loss_flux_dx,
        "negFlux": params["loss_weight_negFlux"] * loss_negFlux,
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
