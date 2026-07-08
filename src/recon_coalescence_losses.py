"""
Loss functions for coalescence surrogate models.
Provides loss computation for different dynamics types (SINDy, NN dzdt, AR, pure AE).
"""

import torch

# Global loss functions
criterion = torch.nn.MSELoss()
divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)


def compute_dzdt_loss(model, batch, params, device):
    """
    Loss for derivative-based dynamics (SINDy or NN dzdt).
    Uses Jacobian-vector products to compute derivatives.

    Args:
        model: ComposedModel with encoder, decoder, and dynamics
        batch: Tuple of (batch_x, batch_dx, batch_M)
               - batch_x: dimensioned DSD
               - batch_dx: time derivative of dimensioned DSD
               - batch_M: total mass (redundant but kept for compatibility)
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_dx, batch_M = batch

    # Forward pass for reconstruction
    # Encoder outputs Z = [z, M], decoder takes Z and outputs dimensioned DSD
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Forward pass for dynamics
    Z = model.encoder(batch_x)
    Z_detached = Z.clone().detach().requires_grad_()
    pred_dZdt = model.dzdt(Z)

    # Compute JVP for encoder and decoder
    _, dZdt = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
    _, pred_dx = torch.func.jvp(model.decoder, (Z_detached,), (pred_dZdt,))

    # Calculate losses
    loss_dx = criterion(pred_dx, batch_dx)
    loss_dz = criterion(pred_dZdt, dZdt)

    # Reconstruction loss in normalized space
    loss_recon = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_recon
        + params["loss_weight_dx"] * loss_dx
        + params["loss_weight_dz"] * loss_dz
    )

    loss_dict = {
        "total": loss,
        "recon": params["loss_weight_recon"] * loss_recon,
        "dx": params["loss_weight_dx"] * loss_dx,
        "dz": params["loss_weight_dz"] * loss_dz,
    }

    return loss, loss_dict


def compute_ar_loss(model, batch, params, device):
    """
    Loss for autoregressive dynamics.

    Args:
        model: ComposedModel with encoder, decoder, and AR dynamics
        batch: Tuple of (batch_X, batch_y, batch_M)
               - batch_X: dimensioned DSD at previous timestep(s)
               - batch_y: dimensioned DSD at next timestep
               - batch_M: total mass
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_X, batch_y, batch_M = batch

    # Reconstruction at t=0
    pred_x_recon = model.decoder(model.encoder(batch_X))

    # Autoregressive prediction
    pred_y, pred_M = model(batch_X, batch_M)

    # Get latent representations for latent loss
    data_Z1 = model.encoder(batch_y)
    pred_Z1 = model.encoder(pred_y)

    # Calculate losses
    loss_dz = criterion(pred_Z1, data_Z1)
    # Loss in normalized space
    loss_dx = divergence(
        torch.log(pred_y / (pred_y.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_y / (batch_y.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )
    loss_recon = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_X / (batch_X.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )

    # Weighted total loss
    loss = (
        params["w_dx"] * loss_dx
        + params["w_recon"] * loss_recon
        + params["w_dz"] * loss_dz
    )

    loss_dict = {
        "total": loss,
        "recon": params["w_recon"] * loss_recon,
        "dx": params["w_dx"] * loss_dx,
        "dz": params["w_dz"] * loss_dz,
    }

    return loss, loss_dict


def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Tuple of (batch_x, batch_dx, batch_M)
               - batch_x: dimensioned DSD
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, _, _ = batch

    # Forward pass for reconstruction
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Calculate losses in normalized space for KL divergence
    loss_kl = divergence(
        torch.log(pred_x_recon / (pred_x_recon.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
        torch.log(batch_x / (batch_x.sum(dim=-1, keepdim=True) + params["tol"]) + params["tol"]),
    )
    # L2 loss in dimensioned space
    loss_l2 = criterion(pred_x_recon, batch_x)

    # Weighted total loss
    loss = loss_kl + params["loss_weight_l2"] * loss_l2

    loss_dict = {
        "total": loss,
        "kl": loss_kl,
        "l2": params["loss_weight_l2"] * loss_l2,
    }

    return loss, loss_dict


def get_loss_function(dynamics_type):
    """
    Get the appropriate loss function for a given dynamics type.

    Args:
        dynamics_type: "sindy", "nn_dzdt", "autoregressive", or "none"

    Returns:
        Loss function callable
    """
    if dynamics_type in ["sindy", "nn_dzdt"]:
        return compute_dzdt_loss
    elif dynamics_type == "autoregressive":
        return compute_ar_loss
    elif dynamics_type == "none":
        return compute_autoencoder_loss
    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
