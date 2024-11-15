import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, ReLU, Sigmoid
import numpy as np
import pysindy as ps

def train_network(training_data, params, val_data=None, device="cpu"):
    # set up the device
    device = torch.device(device)
    print(f"Using {device} device")


    autoencoder_network = MicroAutoEncoder(n_bins=params["input_dim"], n_latent=params["latent_dim"])
    (encoder_weights, encoder_biases) = autoencoder_network.encoder.get_weights()
    (decoder_weights, decoder_biases) = autoencoder_network.decoder.get_weights()
    autoencoder_network.to(device)

    sindy_coeffs_tensor = torch.empty((params["latent_dim"], params["library_size"]), requires_grad=True)
    torch.nn.init.uniform_(sindy_coeffs_tensor)
    training_data["sindy_library"] = torch.empty((params["n_runs"], params["n_time"], params["library_size"]))

    loss, losses = loss_fn(training_data, params, 
                           encoder_weights, encoder_biases, 
                           decoder_weights, decoder_biases,
                           sindy_coeffs_tensor,
                           autoencoder_network,
                           device
                           )
    
    optimizer = torch.optim.Adam([sindy_coeffs_tensor, 
                                  *encoder_weights, *encoder_biases,
                                  *decoder_weights, *decoder_biases
                                  ],
                                  lr = params["learning_rate"]
                                  )
    train_loss = []
    train_losses = {}
    for key in losses.keys():
        train_losses[key] = []
    for epoch in range(params['max_epochs']):
        # just do it on all of the data for now
        optimizer.zero_grad()
        loss, losses = loss_fn(
            training_data, 
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs_tensor, 
            autoencoder_network, 
            device
            )
        
        train_loss.append(loss.detach().numpy().item())
        for key in losses.keys():
            train_losses[key].append(losses[key].detach().numpy().item())

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch%1 == 0:
            print(f'Epoch: {epoch:03d}, Train MSE: {loss.detach().numpy().item():.8f}')
            print([(key, losses[key].detach().numpy().item()) for key in losses.keys()])


    return autoencoder_network, sindy_coeffs_tensor, train_loss, train_losses


def loss_fn(data, 
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs, 
            autoencoder_network, 
            device="cpu"
            ):
    
    # first update the weights & coefficients
    autoencoder_network.encoder.set_weights(encoder_weights, encoder_biases)
    autoencoder_network.decoder.set_weights(decoder_weights, decoder_biases)

    # set up the loss function
    losses = {}
    x = data["x"].to(device)
    dx = data["dx"].to(device)

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

        self.initialize_weights()

    def forward(self,x):
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent) 

        return reconstruction
    
    def initialize_weights(self):
        for network in (self.encoder, self.decoder):
            for layer in network.layers:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                torch.nn.init.uniform_(layer.bias)
    

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
    tmp = torch.zeros_like(x, requires_grad=True)
    l1 = torch.nn.L1Loss()
    loss = l1(x, tmp)
    return loss