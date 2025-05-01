import numpy as np
import torch
import matplotlib.pyplot as plt
from src import data_utils as du
import plotly.graph_objects as go
import plotly.io as pio


def plot_losses(losses, test_losses=None, sub_losses=None, labels=None, title='Training Loss', saveas=None):
    plt.plot(losses, label="total train")
    if test_losses is not None:
        plt.plot(test_losses, label="total test", ls='--')
    if sub_losses is not None:
        for (j, loss) in enumerate(sub_losses):
            plt.plot(loss, label=labels[j])

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()


def plot_reconstructions(model, test_ids, x_test, r_bins_edges, saveas=None):
    (fig, ax) = plt.subplots(ncols=len(test_ids), nrows=2, figsize=(3 * len(test_ids), 6))
    for (i, id) in enumerate(test_ids):
        ax[0][i].step(r_bins_edges, x_test[id, 0])
        ax[1][i].step(r_bins_edges, x_test[id, -1])

        ax[0][i].step(r_bins_edges,
                      model.decoder(model.encoder(torch.Tensor(x_test[id, 0]).reshape(1, 1, -1))).detach().numpy()[
                          0, 0])
        ax[1][i].step(r_bins_edges,
                      model.decoder(model.encoder(torch.Tensor(x_test[id, -1]).reshape(1, 1, -1))).detach().numpy()[
                          0, 0])

        ax[0][i].set_xscale('log')
        ax[1][i].set_xscale('log')
        ax[0][i].set_title(f'Run #{id}')

    ax[0][0].legend(['Data', 'VAE Reconstruction'])
    plt.suptitle('Reconstruction Demo: Out of Sample')
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()


def plot_predictions_AE_AR(model, test_ids, tplt, x_test, m_test, r_bins_edges, n_lag=1, saveas=None):
    (fig, ax) = plt.subplots(ncols=len(test_ids), nrows=len(tplt), figsize=(3 * len(test_ids), 2 * len(tplt)), sharey=True)

    for (i, id) in enumerate(test_ids):
        x0 = x_test[id, :n_lag, :]
        m0 = m_test[id, 0]
        x_pred = np.zeros_like(x_test[id])
        x_pred[:n_lag, :] = x0
        for t in range(n_lag, x_test.shape[1]):
            x_pred[t, :] = model(
                torch.Tensor(x_pred[t - n_lag:t, :]).reshape(-1, n_lag, x_pred[t].shape[0]),
                torch.Tensor([m0]).reshape(1, 1, 1)
            ).detach().numpy()[0][0]

        for (j, t) in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[t, :])

            ax[j][i].set_xscale('log')
            ax[j][i].set_xscale('log')
        ax[0][i].set_title(f'Run #{id}')
        ax[-1][i].set_xlabel('radius (um)')

    for (j, t) in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={t}")
    ax[1][0].legend(['Data', 'Model'])
    plt.suptitle(f'VAE Autoregressive model, lag {n_lag}: Multi time step; out of sample')
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

def plot_latent_trajectories_AR(n_latent, model, x_test, m_test, n_lag=1, saveas=None):
    (fig, ax) = plt.subplots(ncols=n_latent + 1, nrows=2, figsize=(12, 6), sharey=False, sharex=True)
    colors = ['blue', 'orange', 'green', 'pink', 'purple', 'gray']
    time = np.linspace(0, x_test.shape[1] - 1, x_test.shape[1])
    for j in range(x_test.shape[0]):
        x0 = x_test[j, :n_lag, :]
        mj = m_test[j, :]
        z0 = np.array([model.encoder(torch.Tensor(x0[t]).reshape(1, -1)).detach().numpy()[0] for t in range(n_lag)])
        z_pred = np.zeros((x_test.shape[1], n_latent + 1))
        z_enc = np.zeros((x_test.shape[1], n_latent + 1))
        z_enc[:, -1] = mj
        z_enc[:n_lag, :-1] = z0
        z_pred[:n_lag, :-1] = z0
        z_pred[:n_lag, -1] = mj[0]
        for t in range(n_lag, x_test.shape[1]):
            lagged_input = torch.cat(
                (torch.Tensor(z_pred[t - n_lag:t, :-1]).reshape(n_lag * n_latent), torch.Tensor([mj[0]])))
            z_pred[t, :] = model.autoregressor(lagged_input).detach().numpy()
            z_enc[t, :-1] = model.encoder(torch.Tensor(x_test[j, t, :]).reshape(1, -1)).detach().numpy()[0]

        for i in range(n_latent + 1):
            if i < n_latent:
                labeli = f'z{i}'
                color = colors[i]
            else:
                labeli = 'M / dlnr'
                color = colors[-1]
            ax[0][i].plot(time, z_enc[:, i], label=labeli, color=color, alpha=0.5, lw=0.5)
            ax[1][i].plot(time, z_pred[:, i], label=labeli, color=color, alpha=0.5, lw=0.5)
            ax[0][i].set_xlabel('Elapsed time')

    for i in range(n_latent):
        ax[0][i].set_title(f'z{i + 1}')
        ax[0][i].set_xlim([0, x_test.shape[1] - 1])
    ax[0][-1].set_title('mass (rescaled)')
    ax[0][0].set_ylabel('Data')
    ax[1][0].set_ylabel('Model')

    plt.suptitle(f'Autoregressive Z(t), lag {n_lag}')
    plt.tight_layout()
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

def plot_latent_trajectories_SINDy(n_latent, poly_order,
                                   model, x_test, m_test,
                                   dt, time,
                                   x_train, m_train,
                                   plt_dx=True, saveas=None):
    (fig, ax) = plt.subplots(ncols=n_latent + 1, nrows=2, figsize=(12, 6), sharey=False, sharex=True)
    colors = ['blue', 'orange', 'green', 'pink', 'purple', 'gray']

    # compute limits
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((n_latent + 1, 2))
    for il in range(n_latent):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    # compute all else
    sindy_coeffs = model.dzdt.sindy_coeffs.weight
    z_encoded = model.encoder(torch.Tensor(x_test)).detach().numpy()
    dz_encoded = np.gradient(z_encoded, axis=1) / dt
    for j in range(x_test.shape[0]):
        if plt_dx:
            latents_data = np.concatenate([dz_encoded[j], np.zeros((len(time), 1))], axis=-1)
            latents_pred = model.dzdt(torch.Tensor(z_encoded[j]).reshape(1, len(time), -1), torch.Tensor(m_test[j]).reshape(1, -1, 1)).detach().numpy()[0]
            title = "dz/dt"
        else:
            latents_data = np.concatenate([z_encoded[j], m_test[j].reshape(-1, 1)], axis=-1)
            z0 = np.concatenate((z_encoded[j, 0, :], np.array([m_test[j, 0]])), axis=-1)
            latents_pred = du.sindy_simulate(z0, time, sindy_coeffs.float(), poly_order, zlim)
            title = "Z(t)"

        for i in range(n_latent + 1):
            if i < n_latent:
                labeli = f'z{i}'
                color = colors[i]
            else:
                labeli = 'M / dlnr'
                color = colors[-1]
            ax[0][i].plot(time, latents_data[:, i], color=color, alpha=0.5, lw=0.5)
            ax[0][i].set_title(labeli)
            ax[1][i].plot(time, latents_pred[:, i], color=color, alpha=0.5, lw=0.5)
            ax[1][i].set_xlabel('Elapsed time')
    for i in range(n_latent + 1):
        if not plt_dx:
            ax[0][i].set_ylim([zlim[i][0], zlim[i][1]])
            ax[1][i].set_ylim([zlim[i][0], zlim[i][1]])

    ax[0][0].set_ylabel('Data')
    ax[1][0].set_ylabel('Prediction')
    plt.suptitle(title)
    plt.tight_layout()
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

def plot_predictions_AE_SINDy(test_ids, tplt,
                              n_latent, poly_order,
                               model, x_test, m_test,
                               x_train, m_train,
                               r_bins_edges, saveas=None):
    (fig, ax) = plt.subplots(ncols=len(test_ids), nrows=len(tplt), figsize=(3 * len(test_ids), 2 * len(tplt)),
                             sharey=True)

    # compute limits
    z_enc_train = model.encoder(torch.Tensor(x_train)).detach().numpy()
    zlim = np.zeros((n_latent + 1, 2))
    for il in range(n_latent):
        zlim[il][0] = z_enc_train[:, :, il].min()
        zlim[il][1] = z_enc_train[:, :, il].max()
    zlim[-1][0] = m_train.min()
    zlim[-1][1] = m_train.max()

    # compute all else
    sindy_coeffs = model.dzdt.sindy_coeffs.weight
    z_encoded = model.encoder(torch.Tensor(x_test)).detach().numpy()

    for (i, id) in enumerate(test_ids):
        z0 = np.concatenate((z_encoded[id, 0, :], np.array([m_test[id, 0]])), axis=-1)
        latents_pred = du.sindy_simulate(z0, tplt, sindy_coeffs.float(), poly_order, zlim)
        x_pred = model.decoder(torch.Tensor(latents_pred[:, :-1])).detach().numpy()

        for (j, t) in enumerate(tplt):
            ax[j][i].step(r_bins_edges, x_test[id, t, :])
            ax[j][i].step(r_bins_edges, x_pred[j, :])

            ax[j][i].set_xscale('log')
            ax[j][i].set_xscale('log')
        ax[0][i].set_title(f'Run #{id}')
        ax[-1][i].set_xlabel('radius (um)')

    for (j, t) in enumerate(tplt):
        ax[j][0].set_ylabel(f"dmdlnr at t={t}")
    ax[1][0].legend(['Data', 'Model'])
    plt.suptitle(f'AE-SINDy model: Multi time step; out of sample')

    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

def viz_3d_latent_space(model, x_test, time, output_directory, case_name):
    # Save latent space
    lsn = model.encoder(torch.tensor(x_test)).detach().numpy()

    # Plot latent space
    pio.renderers.default = "browser"
    lsnp_data = lsn.reshape((-1, lsn.shape[-1]))
    color_values = (
        np.ones(lsn.shape[:2]) * time
    ).ravel()
    tp_data = np.tile(np.arange(lsn.shape[1]), (lsn.shape[0], 1)).ravel()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=lsnp_data[:, 0],
                y=lsnp_data[:, 1],
                z=lsnp_data[:, 2],
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
    fig.write_html(output_directory + "/" + case_name + "_latent_space.html")
