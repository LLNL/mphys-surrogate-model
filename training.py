import torch
import models
import data_utils as du
from torch.nn import ReLU, Sigmoid, Identity

# Losses
def recon_loss(recon_x, x):
    mseloss = torch.nn.MSELoss()
    return mseloss(recon_x, x)

def l1_loss(x):
    tmp = torch.zeros_like(x, requires_grad=True)
    l1 = torch.nn.L1Loss()
    loss = l1(x, tmp)
    return loss

# Functions for training end-to-end network + SINDy
def train_network_e2e(train_dataloader, params, val_dataloader=None, device="cpu"):
    device = torch.device(device)
    print(f"Using {device} device")

    num_batches = len(train_dataloader)

    autoencoder_network = models.FFNNAutoEncoder(n_bins=params["input_dim"], n_latent=params["latent_dim"])
    (encoder_weights, encoder_biases) = autoencoder_network.encoder.get_weights()
    (decoder_weights, decoder_biases) = autoencoder_network.decoder.get_weights()
    autoencoder_network.to(device)

    sindy_coeffs_tensor = torch.empty((params["latent_dim"], params["library_size"]), requires_grad=True)
    torch.nn.init.uniform_(sindy_coeffs_tensor)
    
    optimizer = torch.optim.Adam([sindy_coeffs_tensor, 
                                  *encoder_weights, *encoder_biases,
                                  *decoder_weights, *decoder_biases
                                  ],
                                  lr = params["learning_rate"]
                                  )
    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)

    train_loss = []
    train_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}
    val_loss = []
    val_losses = {"recon": [], "sindy_z": [], "sindy_x": [], "sindy_reg": []}

    print('TRAINING')
    for epoch in range(params['max_epochs']):
        (epoch_loss, epoch_losses) = train_e2e(optimizer, 
                                               train_dataloader, 
                                               params,
                                               encoder_weights,
                                               encoder_biases,
                                               decoder_weights,
                                               decoder_biases,
                                               sindy_coeffs_tensor,
                                               autoencoder_network,
                                               device)
        train_loss.append(epoch_loss.float())
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e(
                                               train_dataloader, 
                                               params,
                                               encoder_weights,
                                               encoder_biases,
                                               decoder_weights,
                                               decoder_biases,
                                               sindy_coeffs_tensor,
                                               autoencoder_network,
                                               device)
                                               
        val_loss.append(val_epoch_loss.float())
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch%10 == 0:
            print(f'\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss.float():.8f} | Val MSE: {val_epoch_loss.float():.8f}')
            for key in epoch_losses.keys():
                print(f'{key}: {epoch_losses[key]} | {val_epoch_losses[key]}')

        # Check early stopping
        early_stopping(epoch_loss.float())
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    print('\n REFINEMENT')
    ref_params = params
    ref_params['loss_weight_sindy_reg'] = 0.0
    
    for epoch in range(params['refinement_epochs']):
        (epoch_loss, epoch_losses) = train_e2e(optimizer, 
                                               train_dataloader, 
                                               ref_params,
                                               encoder_weights,
                                               encoder_biases,
                                               decoder_weights,
                                               decoder_biases,
                                               sindy_coeffs_tensor,
                                               autoencoder_network,
                                               device)
        
        train_loss.append(epoch_loss.float())
        for key in epoch_losses.keys():
            train_losses[key].append(epoch_losses[key])

        (val_epoch_loss, val_epoch_losses) = test_e2e(
                                               train_dataloader, 
                                               params,
                                               encoder_weights,
                                               encoder_biases,
                                               decoder_weights,
                                               decoder_biases,
                                               sindy_coeffs_tensor,
                                               autoencoder_network,
                                               device)
        val_loss.append(val_epoch_loss.float())
        for key in val_epoch_losses.keys():
            val_losses[key].append(val_epoch_losses[key])

        if epoch%10 == 0:
            print(f'\n Epoch: {epoch:03d}, \n Train MSE: {epoch_loss.float():.8f} | Val MSE: {val_epoch_loss.float():.8f}')
            for key in epoch_losses.keys():
                print(f'{key}: {epoch_losses[key]} | {val_epoch_losses[key]}')

        # Check early stopping
        early_stopping(epoch_loss.float())
        if early_stopping.early_stop:
            print("Training stopped early.")
            break

    return autoencoder_network, sindy_coeffs_tensor, train_loss, train_losses, val_loss, val_losses

def train_e2e(optimizer, train_dataloader, params,
                encoder_weights, encoder_biases,
                decoder_weights, decoder_biases,
                sindy_coeffs_tensor, 
                autoencoder_network, 
                device
                ):
    autoencoder_network.train()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}
    
    for batch, (x_data, dx_data) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x_data = x_data.to(device)
        dx_data = dx_data.to(device)
        
        loss, losses = loss_fn_e2e(
            x_data,
            dx_data,
            params,
            encoder_weights, encoder_biases,
            decoder_weights, decoder_biases,
            sindy_coeffs_tensor, 
            autoencoder_network, 
            device
            )
        
        loss.backward(retain_graph=True)
        optimizer.step()
        
        epoch_loss += loss
        for key in losses.keys():
            epoch_losses[key] += losses[key]
    
    epoch_loss /= len(train_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(train_dataloader)

    return (epoch_loss, epoch_losses)

def test_e2e(val_dataloader, params,
                encoder_weights, encoder_biases,
                decoder_weights, decoder_biases,
                sindy_coeffs_tensor, 
                autoencoder_network, 
                device
                ):
    autoencoder_network.eval()
    epoch_loss = 0.0
    epoch_losses = {"recon": 0.0, "sindy_z": 0.0, "sindy_x": 0.0, "sindy_reg": 0.0}
    
    with torch.no_grad():
        for batch, (x_data, dx_data) in enumerate(val_dataloader):
            x_data = x_data.to(device)
            dx_data = dx_data.to(device)
            
            loss, losses = loss_fn_e2e(
                x_data,
                dx_data,
                params,
                encoder_weights, encoder_biases,
                decoder_weights, decoder_biases,
                sindy_coeffs_tensor, 
                autoencoder_network, 
                device
                )
            
            epoch_loss += loss
            for key in losses.keys():
                epoch_losses[key] += losses[key]
    
    epoch_loss /= len(val_dataloader)
    for key in losses.keys():
        epoch_losses[key] /= len(val_dataloader)

    return (epoch_loss, epoch_losses)

def loss_fn_e2e(x_data,
            dx_data,
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
    x = x_data.to(device)
    dx = dx_data.to(device)

    # reconstruction loss
    x_recon = autoencoder_network(x)
    losses["recon"] = recon_loss(x, x_recon)

    # sindy_dz
    xx = x.clone().detach().requires_grad_()
    z = autoencoder_network.encoder(xx)
    gradient_x = torch.func.vmap(torch.func.vmap(torch.func.jacrev(autoencoder_network.encoder)))(x)
    dz = torch.einsum('abcd, abd->abc', gradient_x, dx)

    sindy_library = du.sindy_library_tensor(z, params["latent_dim"]).to(device)
    sindy_coeffs = sindy_coeffs.to(device)
    dz_sindy = torch.matmul(sindy_library, sindy_coeffs.T)
    losses["sindy_z"] = recon_loss(dz, dz_sindy)

    # sindy_dx
    z = z.clone().detach().requires_grad_()
    gradient_z = torch.func.vmap(torch.func.vmap(torch.func.jacrev(autoencoder_network.decoder)))(z)
    dx_recon = torch.einsum('abcd, abd->abc', gradient_z, dz_sindy)

    losses["sindy_x"] = recon_loss(dx, dx_recon)
    
    # sindy reg
    losses["sindy_reg"] = l1_loss(sindy_coeffs)

    loss = 0.0
    for i, key in enumerate(losses.keys()):
        loss += losses[key] * params["loss_weight_" + key]

    return loss, losses

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")


# Functions for standalone training of network (i.e. no timeseries, not end-to-end)
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    for bindata in dataloader:
        optimizer.zero_grad()
        bindata = bindata.to(device)
        recon = model(bindata.float())
        loss = loss_fn(recon, bindata.float())

        # backprop
        loss.backward()
        optimizer.step()

def test(model, dataloader, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for bindata in dataloader:
            bindata = bindata.to(device)
            recon = model(bindata.float())
            test_loss += loss_fn(recon.float(), bindata.float())
    
    test_loss /= num_batches
    return test_loss