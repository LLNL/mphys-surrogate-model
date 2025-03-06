import numpy as np
import xarray as xr
import torch
import pickle as pkl
import models
import uuid

num_epochs = 1
batch_size = 100
lr = 1e-3

class VAEAutoregressor(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=100,n_latent=10):
        super(VAEAutoregressor, self).__init__()

        self.encoder = models.CNNEncoderVAE(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)
        self.autoregressor = models.Autoregressive(n_bins=n_latent)
        self.decoder = models.CNNDecoder(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)

    def forward(self,x):
        latent = self.encoder(x)
        next_latent = self.autoregressor(latent)
        next_x = self.decoder(next_latent)
        return next_x

# Open dataset
ds_all = xr.open_dataset('box64_train.nc')
x_scale = ds_all['dvdlnr'].max().data
x = ds_all['dvdlnr'].transpose('run','time','mass_bin_idx').to_numpy()
x = x / x_scale
ds_test = xr.open_dataset('box64_test.nc')
x_test = ds_test['dvdlnr'].transpose('run','time','mass_bin_idx').to_numpy()
x_test = x_test / x_scale

# Reshape data
inputs = x[:,:-1,:]
inputs = inputs.reshape([-1, inputs.shape[2]])
outputs = x[:,1:,:]
outputs = outputs.reshape([-1, outputs.shape[2]])

test_inputs = x_test[:,:-1,:]
test_inputs = test_inputs.reshape([-1, test_inputs.shape[2]])
test_outputs = x_test[:,1:,:]
test_outputs = test_outputs.reshape([-1, test_outputs.shape[2]])

# Convert to PyTorch tensors
inputs = torch.Tensor(inputs).reshape((-1, 1, 63))
outputs = torch.Tensor(outputs).reshape(-1, 1, 63)

test_inputs = torch.Tensor(test_inputs).reshape((-1, 1, 63))
test_outputs = torch.Tensor(test_outputs).reshape(-1, 1, 63)

# Initialize the model
model = VAEAutoregressor(n_channels=1, n_bins=63, n_latent=3)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Training loop
# Convert data to batches
train_data = torch.utils.data.TensorDataset(inputs, outputs)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = torch.utils.data.TensorDataset(test_inputs, test_outputs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=x_test.shape[0], shuffle=True)

losses = []
recon_losses = []
dx_losses = []
dz_losses = []

test_losses = []
test_recon_losses = []
test_dx_losses = []
test_dz_losses = []

for epoch in range(num_epochs):
    # train
    for batch_X, batch_y in train_loader:
        # Forward pass
        pred_y = model(batch_X)
        pred_z = model.encoder(batch_X)
        pred_z1 = model.autoregressor(pred_z)
        data_z1 = model.encoder(batch_y)
        pred_x_recon = model.decoder(model.encoder(batch_X))

        loss_dx = criterion(pred_y, batch_y)
        loss_dz = criterion(pred_z1, data_z1)
        loss_recon = criterion(pred_x_recon, batch_X)
        loss = loss_dx + loss_recon + loss_dz

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    recon_losses.append(loss_recon.item())
    dx_losses.append(loss_dx.item())
    dz_losses.append(loss_dz.item())

    # test
    for batch_X, batch_y in test_loader:
        # Forward pass
        pred_y = model(batch_X)
        pred_z = model.encoder(batch_X)
        pred_z1 = model.autoregressor(pred_z)
        data_z1 = model.encoder(batch_y)
        pred_x_recon = model.decoder(model.encoder(batch_X))

        loss_dx = criterion(pred_y, batch_y)
        loss_dz = criterion(pred_z1, data_z1)
        loss_recon = criterion(pred_x_recon, batch_X)
        loss = loss_dx + loss_recon + loss_dz

    test_losses.append(loss.item())
    test_recon_losses.append(loss_recon.item())
    test_dx_losses.append(loss_dx.item())
    test_dz_losses.append(loss_dz.item())

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {losses[-1]:.4f} |  Test Loss: {test_losses[-1]:.4f}")


# Export/save
output_directory = "vae_autoregressor"
case_name = f"lr{lr}_bs{batch_size}_ne{num_epochs}_" + uuid.uuid4().hex

with open(output_directory + '/losses/' + case_name + '.pkl', 'wb') as pickle_file:
    pkl.dump((losses,
            recon_losses,
              dx_losses,
              dz_losses,
              test_losses,
            test_recon_losses,
            test_dx_losses,
            test_dz_losses
            ), pickle_file)

# vae model
torch.save(model.state_dict(), output_directory + '/model/' + case_name + ".pth")
print(f"Saved model and losses as {case_name}")
