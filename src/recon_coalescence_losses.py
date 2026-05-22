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
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, batch_dx, batch_M = batch
    dM = 0.0 * batch_M  # Mass should be conserved (dM/dt = 0)

    # Forward pass for reconstruction
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Forward pass for dynamics
    z = model.encoder(batch_x)
    zz = z.clone().detach().requires_grad_()
    pred_dzM = model.dzdt(z, batch_M)
    pred_dz = pred_dzM[:, :, :-1]  # Latent dynamics
    pred_dM = pred_dzM[:, :, -1:]  # Mass derivative (keep dim for shape match)

    # Compute JVP for encoder and decoder
    _, dz = torch.func.jvp(model.encoder, (batch_x,), (batch_dx,))
    _, pred_dx = torch.func.jvp(model.decoder, (zz,), (pred_dz,))

    # Calculate losses
    loss_dz = criterion(pred_dz, dz) + criterion(pred_dM, dM)
    loss_dx = criterion(pred_dx, batch_dx)
    loss_recon = divergence(
        torch.log(pred_x_recon + params["tol"]),
        torch.log(batch_x + params["tol"]),
    )

    # Weighted total loss
    loss = (
        params["loss_weight_recon"] * loss_recon
        + params["loss_weight_dx"] * loss_dx
        + params["loss_weight_dz"] * loss_dz
    )

    loss_dict = {
        "total": loss,
        "recon": loss_recon,
        "dx": loss_dx,
        "dz": loss_dz,
    }

    return loss, loss_dict


def compute_ar_loss(model, batch, params, device):
    """
    Loss for autoregressive dynamics.

    Args:
        model: ComposedModel with encoder, decoder, and AR dynamics
        batch: Tuple of (batch_X, batch_y, batch_M)
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_X, batch_y, batch_M = batch

    # Reconstruction at t=0
    batch_x = batch_X[:, 0, :]
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Autoregressive prediction
    pred_y, pred_dM = model(batch_X, batch_M)
    pred_y = pred_y[:, 0, :]

    # Get latent representations for latent loss
    data_z1 = model.encoder(batch_y)
    pred_z1 = model.encoder(pred_y)

    # Calculate losses
    loss_dz = criterion(pred_z1, data_z1)
    loss_dx = divergence(
        torch.log(pred_y + params["tol"]),
        torch.log(batch_y + params["tol"]),
    )
    loss_recon = divergence(
        torch.log(pred_x_recon + params["tol"]),
        torch.log(batch_x + params["tol"]),
    )

    # Weighted total loss
    loss = (
        params["w_dx"] * loss_dx
        + params["w_recon"] * loss_recon
        + params["w_dz"] * loss_dz
    )

    loss_dict = {
        "total": loss,
        "recon": loss_recon,
        "dx": loss_dx,
        "dz": loss_dz,
    }

    return loss, loss_dict


def compute_autoencoder_loss(model, batch, params, device):
    """
    Loss for pure autoencoder (no dynamics).
    Used for NNWI models.

    Args:
        model: ComposedModel with encoder and decoder only
        batch: Tuple of (batch_x, batch_dx, batch_M) - only batch_x is used
        params: Parameters dict with loss weights
        device: Compute device

    Returns:
        total_loss, loss_dict
    """
    batch_x, _, _ = batch

    # Forward pass for reconstruction
    pred_x_recon = model.decoder(model.encoder(batch_x))

    # Calculate losses
    loss_kl = divergence(
        torch.log(pred_x_recon + params["tol"]),
        torch.log(batch_x + params["tol"]),
    )
    loss_l2 = criterion(pred_x_recon, batch_x)

    # Weighted total loss
    loss = loss_kl + params["loss_weight_l2"] * loss_l2

    loss_dict = {
        "total": loss,
        "kl": loss_kl,
        "l2": loss_l2,
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
