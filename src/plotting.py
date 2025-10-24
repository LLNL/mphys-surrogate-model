import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wasserstein_distance

from src import data_utils as du

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


def plot_losses(
    losses,
    test_losses=None,
    sub_losses=None,
    labels=None,
    title="Training Loss",
    saveas=None,
):
    """
    Plot training and test losses over epochs.

    This function visualizes the evolution of loss values during model training.
    It can plot total training loss, total test loss, and any number of sub-losses
    (such as regularization or reconstruction losses).

    :param losses: Sequence of total training loss values per epoch.
    :param test_losses: Optional sequence of total test loss values per epoch.
    :param sub_losses: Optional list of sequences, each representing a sub-loss per epoch.
    :param labels: Optional list of labels for sub-losses.
    :param title: Title for the plot.
    :param saveas: Optional path to save the figure.
    :return: The matplotlib figure object for further manipulation.
    """

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")

    # Plot losses
    ax.plot(losses, label="total train")
    if test_losses is not None:
        ax.plot(test_losses, label="total test", ls="--")
    if sub_losses is not None:
        for j, loss in enumerate(sub_losses):
            ax.plot(loss, label=labels[j])

    # Accoutrements
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_reconstructions(
    model, test_ids, x_test, r_bins_edges, t_plt=[0, -1], saveas=None
):
    """
    Plot AE reconstructions of DSDs for selected test members and time steps.

    This function compares the actual DSDs to the autoencoder (AE) reconstructions
    for specified test member indices and time steps.

    :param model: Trained autoencoder model.
    :param test_ids: List of indices for test set members to plot.
    :param x_test: Test set DSD data array.
    :param r_bins_edges: Bin edges for DSD radius.
    :param t_plt: List of time indices to plot for each member.
    :param saveas: Optional path to save the figure.
    :return: The matplotlib figure object for further manipulation.
    """
    # Set up figure
    (fig, ax) = plt.subplots(
        nrows=len(t_plt),
        ncols=len(test_ids),
        figsize=(3 * len(test_ids), 2 * len(t_plt)),
        layout="constrained",
    )

    # Plot reconstruction for each test ID for multiple times
    for i, id in enumerate(test_ids):
        for j, t in enumerate(t_plt):
            ax[j][i].step(r_bins_edges, x_test[id, t])
            ax[j][i].step(
                r_bins_edges,
                model.decoder(
                    model.encoder(torch.Tensor(x_test[id, t]).reshape(1, 1, -1))
                )
                .detach()
                .numpy()[0, 0],
            )

            ax[j][i].set_xscale("log")
            ax[j][i].set_ylabel("PSD")
            ax[j][i].set_xlabel("r (m)")
        ax[0][i].set_title(f"Run #{id}")

    # Accoutrements
    ax[0][0].legend(["Data", "AE Reconstruction"])
    fig.suptitle("Reconstruction Demo: Out of Sample")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_predictions_AE_AR(
    model, test_ids, dsd_time, tplt, x_test, m_test, r_bins_edges, n_lag=1, saveas=None
):
    """
    Plot predictions of DSDs using an AE-AR model for selected members and time steps.

    This function visualizes the model's predictions for DSD evolution using an
    autoencoder-autoregressive (AE-AR) approach, comparing predicted and true DSDs.

    :param model: Trained AE-AR model.
    :param test_ids: List of indices for test set members to plot.
    :param dsd_time: Array of time values for DSD evolution.
    :param tplt: List of time indices to plot for each member.
    :param x_test: Test set DSD data array.
    :param m_test: Test set mass data array.
    :param r_bins_edges: Bin edges for DSD radius.
    :param n_lag: Number of lagged time steps used for prediction.
    :param saveas: Optional path to save the figure.
    :return: The matplotlib figure object for further manipulation.
    """
    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(3 * len(test_ids), 2 * len(tplt)),
        sharey=True,
        layout="constrained",
    )
    model.eval()

    # Plot predictions for AE-AR model for multiple time steps
    for i, id in enumerate(test_ids):
        x0 = x_test[id, :n_lag, :]
        m0 = m_test[id, 0]
        x_pred = np.zeros_like(x_test[id])
        x_pred[:n_lag, :] = (
            model.decoder(model.encoder(torch.Tensor(x0))).detach().numpy()
        )
        for t in range(n_lag, x_test.shape[1]):
            x_pred[t, :] = (
                model(
                    torch.Tensor(x_pred[t - n_lag : t, :]).reshape(
                        -1, n_lag, x_pred[t].shape[0]
                    ),
                    torch.Tensor([m0]).reshape(1, 1, 1),
                )
                .detach()
                .numpy()[0][0]
            )

        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[t, :])

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Run #{id}")
        ax[-1][i].set_xlabel("radius (um)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(
        f"VAE Autoregressive model, lag {n_lag}: Multi time step; out of sample"
    )

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_latent_trajectories(
    n_latent,
    dsd_time,
    z_pred,
    z_data,
    saveas=None,
    n_samples=None,
):
    """
    Plot latent trajectories for test set DSDs and model predictions.

    This function visualizes the evolution of latent variables and rescaled mass
    for both the data and model predictions across time.

    :param n_latent: Number of latent dimensions.
    :param dsd_time: Array of time values for DSD evolution.
    :param z_pred: Predicted latent trajectories from the model.
    :param z_data: True latent trajectories from the data.
    :param saveas: Optional path to save the figure.
    :param n_samples: Optional number of samples to plot.
    :return: The matplotlib figure object for further manipulation.
    """
    # Set up figure
    (fig, ax) = plt.subplots(
        nrows=2,
        ncols=n_latent + 1,
        figsize=(3 * (n_latent + 1), 6),
        sharey=False,
        sharex=True,
        layout="constrained",
    )
    colors = ["blue", "orange", "green", "pink", "purple", "gray"]

    if n_samples is None:
        n_samples = z_pred.shape[0]

    for j in range(n_samples):
        for i in range(n_latent + 1):
            if i < n_latent:
                labeli = f"z{i}"
                color = colors[i]
            else:
                labeli = "M / dlnr"
                color = colors[-1]
            ax[0][i].plot(
                dsd_time,
                z_data[j, :, i],
                label=labeli,
                color=color,
                alpha=min(1, 150 / n_samples),
                lw=0.5,
            )
            ax[1][i].plot(
                dsd_time,
                z_pred[j, :, i],
                label=labeli,
                color=color,
                alpha=min(1, 150 / n_samples),
                lw=0.5,
            )
            ax[-1][i].set_xlabel("Elapsed time (s)")

    # Accoutrements
    for i in range(n_latent):
        ax[0][i].set_title(f"z{i + 1}")
        ax[0][i].set_xlim([0, dsd_time.max()])
    ax[0][-1].set_title("mass (rescaled)")
    ax[0][0].set_ylabel("Data")
    ax[1][0].set_ylabel("Model")
    fig.suptitle(f"Test set predicted Z(t)")
    plt.tight_layout()

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_latent_trajectories_heatmap(
    n_latent, dsd_time, z_pred, z_data, saveas=None, n_samples=None
):
    """
    Visualize latent trajectories as heatmaps for test set DSDs and predictions.

    This function creates heatmaps of the latent variable distributions and rescaled mass
    over time, optionally overlaying individual sample trajectories.

    :param n_latent: Number of latent dimensions.
    :param dsd_time: Array of time values for DSD evolution.
    :param z_pred: Predicted latent trajectories from the model.
    :param z_data: True latent trajectories from the data.
    :param saveas: Optional path to save the figure.
    :param n_samples: Optional number of samples to overlay.
    :return: The matplotlib figure object for further manipulation.
    """
    # Set up figure
    (fig, ax) = plt.subplots(
        nrows=2,
        ncols=n_latent + 1,
        figsize=(3 * (n_latent + 1), 6),
        sharey=False,
        sharex=True,
        layout="constrained",
    )
    cmap = plt.colormaps["viridis"]
    t_all = np.broadcast_to(dsd_time, (z_pred.shape[0], len(dsd_time)))

    for i in range(n_latent + 1):
        h_data, tedges, zedges = np.histogram2d(
            t_all.flatten(), z_data[:, :, i].flatten(), bins=[len(dsd_time), 20]
        )
        h_pred, tedges, zedges = np.histogram2d(
            t_all.flatten(), z_pred[:, :, i].flatten(), bins=[len(dsd_time), 20]
        )
        ax[0][i].pcolormesh(
            tedges,
            zedges,
            h_data.T,
            cmap=cmap,
        )
        ax[1][i].pcolormesh(
            tedges,
            zedges,
            h_pred.T,
            cmap=cmap,
        )
        ax[-1][i].set_xlabel("Elapsed time (s)")
        ax[0][i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[1][i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if n_samples is not None:
        for j in range(n_samples):
            for i in range(n_latent + 1):
                ax[0][i].plot(dsd_time, z_data[j, :, i], color="w", lw=1)
                ax[1][i].plot(
                    dsd_time,
                    z_pred[j, :, i],
                    color="w",
                    lw=1,
                )

    # Accoutrements
    for i in range(n_latent):
        ax[0][i].set_title(f"$z_{i + 1}$")
        ax[0][i].set_xlim([0, dsd_time.max()])
    ax[0][-1].set_title("$M$ (rescaled)")
    ax[0][0].set_ylabel("Data")
    ax[1][0].set_ylabel("Model")
    fig.suptitle(f"Test set predicted Z(t)")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_predictions_dzdt(
    test_ids,
    tplt,
    n_latent,
    model,
    dsd_time,
    x_test,
    m_test,
    x_train,
    m_train,
    r_bins_edges,
    saveas=None,
):
    """
    Plot DSD predictions using ODE integration in latent space for selected test members
    for the dzdt model.

    This function simulates the latent space evolution using the dzdt model's found ODE
    and visualizes the resulting DSD predictions compared to the true data.

    :param test_ids: List of indices for test set members to plot.
    :param tplt: List of time indices to plot for each member.
    :param n_latent: Number of latent dimensions.
    :param model: Trained latent ODE model.
    :param dsd_time: Array of time values for DSD evolution.
    :param x_test: Test set DSD data array.
    :param m_test: Test set mass data array.
    :param x_train: Training set DSD data array.
    :param m_train: Training set mass data array.
    :param r_bins_edges: Bin edges for DSD radius.
    :param saveas: Optional path to save the figure.
    :return: The matplotlib figure object for further manipulation.
    """
    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(test_ids),
        nrows=len(tplt),
        figsize=(3 * len(test_ids), 2 * len(tplt)),
        sharey=True,
    )

    # Compute limits
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((n_latent + 1, 2))
    for il in range(n_latent):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    # Compute all else
    z_encoded = model.encoder(torch.Tensor(x_test)).detach().numpy()
    for i, id in enumerate(test_ids):
        z0 = np.concatenate((z_encoded[id, 0, :], np.array([m_test[id, 0]])), axis=-1)
        latents_pred = du.simulate(z0, dsd_time[tplt], model.dzdt, zlim)
        x_pred = model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()

        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[j, :])

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"Run #{id}")
        ax[-1][i].set_xlabel("radius (um)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(f"Multi time step; out of sample")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def viz_3d_latent_space(model, x_test, time, saveas=None):
    """
    Visualize the autoencoder latent space in 3D for the test set.

    This function creates a 3D scatter plot of the latent representations for the test set
    DSDs, colored by time. The output is an interactive html file that must
    be opened in a browser.

    :param model: Trained autoencoder model.
    :param x_test: Test set DSD data array.
    :param time: Time values corresponding to each sample.
    :param saveas: Optional path to save the figure (as HTML).
    :return: The plotly figure object for further manipulation.
    """
    # get latent space
    lsn = model.encoder(torch.Tensor(x_test)).detach().numpy()

    # Plot latent space
    pio.renderers.default = "browser"
    lsnp_data = lsn.reshape((-1, lsn.shape[-1]))
    x = lsnp_data[:, 0]
    if lsnp_data.shape[1] < 2:
        y = lsnp_data[:, 0] * 0.0
    else:
        y = lsnp_data[:, 1]
        if lsnp_data.shape[1] < 3:
            z = lsnp_data[:, 0] * 0.0
        else:
            z = lsnp_data[:, 2]
    color_values = (np.ones(lsn.shape[:2]) * time).ravel()
    tp_data = np.tile(np.arange(lsn.shape[1]), (lsn.shape[0], 1)).ravel()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name="",
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale="Viridis",
                    opacity=0.7,
                    colorbar=dict(title="Time"),
                    line=dict(color="white", width=0.0),
                ),
                hovertemplate="LD1: %{x:.4f}<br>"
                + "LD2: %{y:.4f}<br>"
                + "LD3: %{z:.4f}<br>"
                + "Test Member: %{marker.color:d}<br>"
                + "Time Step: %{customdata}<br>",
                customdata=tp_data,
            )
        ]
    )
    fig.update_layout(
        title="Autoencoder Latent Space",
        width=1200,
        height=900,
        scene=dict(
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            zaxis_title="Latent Dim 3",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # Optional save
    if saveas is not None:
        fig.write_html(saveas)

    # Return fig for further manipulation
    return fig


def plot_full_testset_performance_recon(model, x_test, tol, saveas=None):
    """
    Plot performance metrics for model predictions across the full test set.

    This function visualizes the KL divergence, Wasserstein distance, and total mass
    difference for model predictions compared to true DSDs, for all test set members
    and time steps.

    :param test_kl: Array of KL divergence values for predictions.
    :param test_wass: Array of Wasserstein distance values for predictions.
    :param test_mass_diff: Array of total mass differences for predictions.
    :param saveas: Optional path to save the figure.
    :param figsize: Optional figure size.
    :return: The matplotlib figure object for further manipulation.
    """

    # Extra vars
    n_test = x_test.shape[0]
    n_timesteps = x_test.shape[1]
    divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    # Determine best and worst performing members
    test_preds = model.decoder(model.encoder(torch.Tensor(x_test)))
    test_kl = np.zeros(x_test.shape[0:2])
    test_wass = np.zeros(x_test.shape[0:2])
    for nm in range(n_test):
        for nt in range(n_timesteps):
            pred_dist = test_preds[nm, nt]
            true_dist = torch.Tensor(x_test[nm, nt]).reshape(1, 1, -1)
            test_kl[nm, nt] = divergence(
                torch.log(pred_dist + tol),
                torch.log(true_dist + tol),
            )
            test_wass[nm, nt] = wasserstein_distance(
                pred_dist.detach().numpy().ravel(), true_dist.detach().numpy().ravel()
            )

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(34, 5), layout="constrained")
    # ---
    ax = axes[0]
    klm = ax.matshow(np.log10(test_kl.T), vmin=-5, vmax=-2)
    fig.colorbar(
        klm,
        ax=ax,
        location="top",
        label=f"log10(KL Divergence) (Mean={np.mean(np.log10(test_kl)):.2f})",
        extend="both",
    )
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[1]
    wsm = ax.matshow(test_wass.T, vmin=0.0005, vmax=0.008)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Wasserstein Distance (Mean={np.mean(test_wass):.2e})",
        extend="both",
    )
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_full_testset_performance_pred(
    test_kl, test_wass, test_mass_diff, saveas=None, figsize=None
):
    """
    Plot performance metrics for model predictions across the full test set.

    This function visualizes the KL divergence, Wasserstein distance, and total mass
    difference for model predictions compared to true DSDs, for all test set members
    and time steps.

    :param test_kl: Array of KL divergence values for predictions.
    :param test_wass: Array of Wasserstein distance values for predictions.
    :param test_mass_diff: Array of total mass differences for predictions.
    :param saveas: Optional path to save the figure.
    :param figsize: Optional figure size.
    :return: The matplotlib figure object for further manipulation.
    """
    # Plot
    if figsize is None:
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=(34, 8), layout="constrained"
        )
    else:
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=figsize, layout="constrained"
        )
    # ---
    ax = axes[0]
    klm = ax.matshow(np.log10(test_kl.T), vmin=-5, vmax=-2)
    fig.colorbar(
        klm,
        ax=ax,
        location="top",
        label=f"log10(KL Divergence) (Mean={np.mean(np.log10(test_kl)):.2f})",
        extend="both",
    )
    print(f"KL Divergence (Mean={np.mean(test_kl):.2e})")
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[1]
    wsm = ax.matshow(test_wass.T, vmin=0.0005, vmax=0.008)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Wasserstein Distance (Mean={np.mean(test_wass):.2e})",
        extend="both",
    )
    print(f"Wasserstein Distance (Mean={np.mean(test_wass):.2e})")
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")
    # ---
    ax = axes[2]
    wsm = ax.matshow(test_mass_diff.T, vmin=-0.05, vmax=0.05)
    fig.colorbar(
        wsm,
        ax=ax,
        location="top",
        label=f"Total Mass Difference (MAE={np.mean(np.abs(test_mass_diff)):.2e})",
        extend="both",
    )
    print(f"Total Mass Difference (MAE={np.mean(np.abs(test_mass_diff)):.2e})")
    ax.set_xlabel(f"Test Member")
    ax.set_ylabel(f"Time")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig


def plot_testset_quantiles_pred(
    x_test,
    test_preds,
    test_metric,
    tplt,
    dsd_time,
    r_bins_edges,
    qtiles=(0, 0.25, 0.5, 0.75, 0.9999),
    saveas=None,
):
    """
    Plot DSD predictions for test set members at specified performance percentiles.

    This function selects test set members based on performance quantiles (using a
    specified metric) and visualizes their true and predicted DSDs for selected time steps.

    :param x_test: Test set DSD data array.
    :param test_preds: Predicted DSDs from the model.
    :param test_metric: Metric array used for quantile selection.
    :param tplt: List of time indices to plot for each member.
    :param dsd_time: Array of time values for DSD evolution.
    :param r_bins_edges: Bin edges for DSD radius.
    :param qtiles: Tuple of quantiles to plot.
    :param saveas: Optional path to save the figure.
    :return: The matplotlib figure object for further manipulation.
    """
    n_test = x_test.shape[0]
    test_metric_timemean = np.mean(test_metric, axis=1)
    tm_argsort = np.argsort(-test_metric_timemean)
    qtile_idx = (np.array(qtiles) * n_test).astype(int)
    qtile_mems = tm_argsort[qtile_idx]
    print(qtile_mems)

    # Set up figure
    (fig, ax) = plt.subplots(
        ncols=len(qtile_mems),
        nrows=len(tplt),
        figsize=(3 * len(qtile_mems), 2 * len(tplt)),
        sharey=True,
    )

    for i, id in enumerate(qtile_mems):
        for j, t in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, test_preds[id, t, :].detach().numpy())

            ax[j][i].set_xscale("log")
            ax[j][i].set_xscale("log")
        ax[0][i].set_title(f"{qtiles[i] * 100}th percentile")
        ax[-1][i].set_xlabel("radius (m)")

    # Accoutrements
    for j, t in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={dsd_time[t]}")
    ax[1][0].legend(["Data", "Model"])
    fig.suptitle(f"Multi time step; out of sample; percentiles")

    # Optional save
    if saveas is not None:
        fig.savefig(saveas)

    # Return fig for further manipulation
    return fig
