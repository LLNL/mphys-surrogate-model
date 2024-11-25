import models
import data_utils as du
import training
import torch
import numpy as np
from tqdm import tqdm

# parameters
batch_size = 500
latent_try = np.array([2, 3, 5, 8, 13])
epochs = 1000
init_lr = 1e-3
weight_decay = 1e-3
output_path = "./models_unscaled/"

# get data
train_dataloader, test_dataloader, val_dataloader = du.create_dataloader("../pysdm_data/", batch_size)

# set up the device
device = torch.device(
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
    )
torch.backends.cudnn.benchmark = (
	True  # Used if inputs & model are constant, dropout?
    )
print(f"Using {device} device")

train_mse_recs = np.zeros((latent_try.size, epochs))
val_mse_recs = np.zeros((latent_try.size, epochs))

for il, n_latent in enumerate(latent_try):
	# define model
	model = models.CNNAutoEncoder(n_latent=n_latent)
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
	criterion = torch.nn.MSELoss()

	for epoch in tqdm(range(0,epochs)):
	    
		mod = training.train(model, train_dataloader, training.recon_loss, optimizer, device)
		train_mse = training.test(model, train_dataloader, training.recon_loss, device)
		val_mse = training.test(model, val_dataloader, training.recon_loss, device)

		train_mse_recs[il, epoch] = train_mse
		val_mse_recs[il, epoch] = val_mse

		if epoch%20 == 0:
			print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.8e}, Val. MSE: {val_mse:.8e}')

	tmp_trainloss = "VAE_z{}_mse_{}_epochs_losses.npz".format(n_latent, epochs)
	np.savez(output_path + tmp_trainloss, train_loss=train_mse_recs, val_loss=val_mse_recs)

	torch.save(model.state_dict(), output_path +  "model" + str(n_latent) + ".pth")
	print(f"Saved VAE with {n_latent} latent variables to model{n_latent}.pth")
