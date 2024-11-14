import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, ReLU, Sigmoid
import numpy as np
import pysindy as ps

def train_network(training_data, params, val_data=None, device="cpu"):
    # set up network
    device = torch.device(device)
    autoencoder_network = MicroAutoEncoder(n_bins=params["input_dim"], n_latent=params["latent_dim"])
    (encoder_weights, encoder_biases) = autoencoder_network.encoder.get_weights()
    (decoder_weights, decoder_biases) = autoencoder_network.decoder.get_weights()
    autoencoder_network.to(device)

    sindy_model = ps.SINDy(
        feature_library = custom_library
    )
    sindy_model.fit(training_data["z"], training_data["t"])
    sindy_coeffs_tensor = torch.Tensor(sindy_model.coefficients())
    training_data["sindy_library"] = torch.empty((params["n_runs"], params["n_time"], params["library_size"]))

    loss, losses = loss_fn(training_data, params, 
                           encoder_weights, encoder_biases, 
                           decoder_weights, decoder_biases,
                           sindy_coeffs_tensor,
                           autoencoder_network,
                           sindy_model,
                           device)
    
    optimizer = torch.optim.Adam([sindy_coeffs_tensor, 
                                  *encoder_weights, *encoder_biases,
                                  *decoder_weights, *decoder_biases
                                  ]
                                  )
    for i in range(params['max_epochs']):
        # just do it on all of the data for now
        optimizer.zero_grad()
        loss, _ = loss_fn(
            training_data, 
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs_tensor, 
            autoencoder_network, 
            sindy_model, 
            device)
        loss.backward(retain_graph=True)
        optimizer.step()

    return autoencoder_network, sindy_coeffs_tensor, loss, losses


def loss_fn0(data, 
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs_tensor, 
            autoencoder_network, 
            sindy_model, 
            device,
            batch_size=None
            ):
    if batch_size is None:
        batch_size = data["x"].shape[0] * data["x"].shape[1]

    # first update the weights & coefficients
    autoencoder_network.encoder.set_weights(encoder_weights, encoder_biases)
    autoencoder_network.decoder.set_weights(decoder_weights, decoder_biases)
    sindy_coeffs = sindy_coeffs_tensor.detach().numpy()
    sindy_model.optimizer.coef_ = sindy_coeffs

    # set up the loss function
    losses = {}

    # precompute stuff
    x = torch.Tensor(data["x"]) #x = data["x"]
    dx = torch.Tensor(data["dx"]) #dx = data["dx"]
    z = np.zeros((params["n_runs"], params["n_time"], params["latent_dim"]))
    dz = np.zeros_like(z)
    x_recon = np.zeros_like(x)
    dx_decode = np.zeros_like(x)

    for i in range(params["n_runs"]):  # probably a way to clean this up later...
        for j in range(params["n_time"]):
            x_in = np.reshape(x[i, j, :], (1, params["input_dim"]))
            dx_in = np.reshape(dx[i, j, :], (1, params["input_dim"]))
            x_tensor = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)
            dx_tensor = torch.tensor(dx_in, dtype=torch.float32)

            x_recon_tensor = autoencoder_network(x_tensor)
            x_recon[i, j, :] = x_recon_tensor.detach().numpy()
        
            z_tensor = autoencoder_network.encoder(x_tensor)
            grad_x = torch.empty((params["latent_dim"], params["input_dim"]), dtype=torch.float32)
            for il in range(params["latent_dim"]):
                zl = z_tensor[0][il]
                zl.backward(retain_graph = True)
                grad_x[il, :] = x_tensor.grad
            dz_tensor = torch.matmul(grad_x, dx_tensor[0])
            z[i, j, :] = z_tensor.detach().numpy()
            dz[i, j, :] = dz_tensor.detach().numpy()

            grad_z = torch.empty((params["input_dim"], params["latent_dim"]), dtype=torch.float32)
            z_vae = torch.tensor(z_tensor, dtype=torch.float32, requires_grad=True)
            x_recon_tensor = autoencoder_network.decoder(z_vae)
            for ib in range(params["input_dim"]):
                xb = x_recon_tensor[0][ib]
                xb.backward(retain_graph = True)
                grad_z[ib, :] = z_vae.grad
            dz_sindy = torch.tensor(sindy_model.predict(z_vae.detach().numpy()), dtype=torch.float32)
            dx_sindy = torch.matmul(grad_z, dz_sindy[0])
            dx_decode[i, j, :] = dx_sindy.detach().numpy()
    
    # reconstruction
    data["x_recon"] = x_recon
    losses["recon"] = np.linalg.norm(data["x"] - data["x_recon"])

    # sindy dz
    data["z"] = z
    data["dz"] = dz
    data["dz_sindy"] = sindy_model.predict(z)
    losses["sindy_z"] = np.linalg.norm(dz - data["dz_sindy"])

    # sindy dx
    data["dx_decode"] = dx_decode
    losses["sindy_x"] = np.linalg.norm(dx - data["dx_decode"])

    # sindy regularization loss
    losses["sindy_reg"] = np.mean(np.abs(sindy_model.coefficients()))
    
    loss = 0.0
    for i, key in enumerate(losses.keys()):
        loss += losses[key] * params["loss_weight_" + key]

    return loss, losses

def loss_fn(data, 
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs, 
            autoencoder_network, 
            sindy_model,
            device,
            batch_size=None
            ):
    if batch_size is None:
        batch_size = data["x"].shape[0] * data["x"].shape[1]

    # first update the weights & coefficients
    autoencoder_network.encoder.set_weights(encoder_weights, encoder_biases)
    autoencoder_network.decoder.set_weights(decoder_weights, decoder_biases)

    # set up the loss function
    losses = {}
    x = data["x"]
    dx = data["dx"]

    # reconstruction loss
    x_recon = autoencoder_network(x)
    data["x_recon"] = x_recon
    losses["recon"] = recon_loss(x, x_recon)

    # sindy_dz
    z = autoencoder_network.encoder(x)
    gradient_x = torch.empty((params["n_runs"], params["n_time"],params["latent_dim"], params["input_dim"]))
    for il in range(params["latent_dim"]):
        unit_vec = torch.zeros_like(z)
        unit_vec[:, :, il] = 1.0
        z.backward(unit_vec, retain_graph=True)
        gradient_x[:, :, il, :] = x.grad
    dz = torch.einsum('abcd, abd->abc', gradient_x, data["dx"])
    data["z"] = z
    data["dz"] = dz
    data["sindy_library"] = sindy_library_tensor(z, params["latent_dim"], data["sindy_library"])
    data["dz_sindy"] = torch.matmul(data["sindy_library"], sindy_coeffs.T)
    losses["sindy_z"] = recon_loss(dz, data["dz_sindy"])

    # sindy_dx
    gradient_z = torch.empty(((params["n_runs"], params["n_time"],params["input_dim"], params["latent_dim"])))
    z = z.clone().detach().requires_grad_()
    x_recon = autoencoder_network.decoder(z)
    for ib in range(params["input_dim"]):
        unit_vec = torch.zeros_like(x)
        unit_vec[:, :, ib] = 1.0
        x_recon.backward(unit_vec, retain_graph=True)
        gradient_z[:, :, ib, : ] = z.grad
    dx_recon = torch.einsum('abcd, abd->abc', gradient_z, data["dz_sindy"])
    data["dx_sindy_recon"] = dx_recon
    losses["sindy_x"] = recon_loss(dx, data["dx_sindy_recon"])
    
    # sindy reg
    losses["sindy_reg"] = l1_loss(sindy_coeffs)

    loss = 0.0
    for i, key in enumerate(losses.keys()):
        loss += losses[key] * params["loss_weight_" + key]

    return loss, losses


# SINDY functions
library_functions = [
    lambda: 1,
    lambda x: x,
]
library_function_names = [
    lambda: "1",
    lambda x: x,
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)


# Autoencoder
class CNNEncoderVAE(torch.nn.Module):
    def __init__(self,n_bins=127,n_latent=3):
        super(CNNEncoderVAE, self).__init__()
        self.n_bins = n_bins
        self.layer1 = Linear(n_bins, int(n_bins / 2))
        self.activation1 = ReLU()
        self.layer2 = Linear(int(n_bins / 2), int(n_bins / 4))
        self.activation2 = ReLU()
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 8))
        self.activation3 = ReLU()
        self.layer4 = Linear(int(n_bins / 8), n_latent)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        
        return x
    
    def get_weights(self):
        weights = []
        biases = []
        for (i, layer) in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

class CNNDecoder(torch.nn.Module):
    def __init__(self,n_bins=128,n_latent=3):
        super(CNNDecoder, self).__init__()

        self.n_bins = n_bins
        self.layer1 = Linear(n_latent, int(n_bins / 8))
        self.layer2 = Linear(int(n_bins / 8), int(n_bins / 4))
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 2))
        self.layer4 = Linear(int(n_bins / 2), n_bins)
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activation3 = ReLU()
        self.activation4 = Sigmoid()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        
        return x
    
    def get_weights(self):
        weights = []
        biases = []
        for (i, layer) in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

class MicroAutoEncoder(torch.nn.Module):
    def __init__(self,n_bins=100,n_latent=10):
        super(MicroAutoEncoder, self).__init__()

        self.encoder = CNNEncoderVAE(n_bins=n_bins,n_latent=n_latent)
        self.decoder = CNNDecoder(n_bins=n_bins,n_latent=n_latent)

    def forward(self,x):
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent) 

        return reconstruction
    

def sindy_library_tensor(z, latent_dim, current_library):
    # not implemented for order 2 and higher terms
    assert current_library.size()[2] == latent_dim + 1 
    new_library = torch.zeros_like(current_library)

    # i = 0: constant
    new_library[:, :, 0] = 1.0

    # i = 1:nl + 1 -> first order
    new_library[:, :, 1:] = z

    return new_library

def recon_loss(recon_x, x):
    mseloss = torch.nn.MSELoss()
    return mseloss(recon_x, x)

def l1_loss(x):
    tmp = torch.zeros_like(x)
    l1 = torch.nn.L1Loss()
    return l1(x, tmp)