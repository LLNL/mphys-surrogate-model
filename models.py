import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import Linear as Lin, ReLU, Sigmoid, ConstantPad1d

class BinDataset(Dataset):
    def __init__(self, data):
        self.bin0 = data

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :, :]

def create_dataloader(filepath, bs):
    ds = xr.open_mfdataset(filepath + "*.nc", combine='nested', concat_dim='run')
    r_bins_edges = np.logspace(np.log10(0.1 * 1e-6), np.log10(10 * 1e-3), 101, endpoint=True,)
    dr = r_bins_edges[1:] - r_bins_edges[0:-1]
    dlnr = np.log(r_bins_edges[1:]) - np.log(r_bins_edges[0:-1])

    rhow = 1000.0 # kg / m^3

    # flatten the data
    data = ds.stack(case=("run","time_save_spec","height"))
    nc = data['case'].size
    nb = data['wet_spectrum_bin_index'].size

    bin0 = np.zeros((nc, 2, nb))
    bin0[:, 0, :] = data['wet spectrum'].to_numpy().T
    bin0[:, 1, :] = data['dvdlnr'].to_numpy().T

    # data QC
    N_threshold = [0.1 * 1e6,  1e13] #0.1 - 10,000 / cm^3
    M_threshold = (1e-9, 1e16) #0.01 - 10 g / m^3
    casemask = np.zeros(bin0.shape[0])
    binsums = np.sum(bin0, axis=2)
    for i in range(0, bin0.shape[0]):
        if np.any(bin0[i, 0, :] > N_threshold[1]) or np.any(bin0[i, 1, :] > M_threshold[1]):
            casemask[i] = False
        elif binsums[i, 0] < N_threshold[0] or binsums[i, 1] < M_threshold[0]:
            casemask[i] = False
        else:
            casemask[i] = True
    bin0 = bin0[casemask == 1.0, :]

    # Normalization?
    binsums = np.sum(bin0, axis=2)
    momscales = np.max(binsums, axis=0)
    bin0[:, 0, :] /= momscales[0]
    bin0[:, 1, :] /= momscales[1]

    # mean distributions
    binmean = bin0.mean(axis=0)

    ncase, _, nb = bin0.shape
    print(f"Found {ncase} samples out of {nc}")

    # Test-train split
    quart = int(bin0.shape[0]/4)

    idx = np.arange(0,bin0.shape[0])
    np.random.seed(42)
    np.random.shuffle(idx)
    trainidx = idx[0:quart*2]
    validx = idx[quart*2:quart*3]
    testidx = idx[quart*3:]
    print("train ",quart*2,"val ",quart,"test ",quart)

    bin0_train = bin0[trainidx,:]
    bin0_val = bin0[validx,:]
    bin0_test = bin0[testidx,:]

    traindataset = BinDataset(bin0_train)
    valdataset = BinDataset(bin0_val)
    testdataset = BinDataset(bin0_test)

    train_dataloader = DataLoader(traindataset, batch_size=bs)
    test_dataloader = DataLoader(testdataset, batch_size=bs)
    val_dataloader = DataLoader(valdataset, batch_size=bs)

    return (train_dataloader, test_dataloader, val_dataloader)

def recon_loss(recon_x, x):
    mseloss = torch.nn.MSELoss()
    return mseloss(recon_x, x)

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

class CNNEncoderVAE(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=35,n_latent=10):
        super(CNNEncoderVAE, self).__init__()
        self.n_bins = n_bins
        self.conv1 = Conv1d(in_channels=2,out_channels=4,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.conv2 = Conv1d(in_channels=4,out_channels=8,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = Conv1d(in_channels=8,out_channels=4,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()

        
        self.fc_mu = Lin(48,n_latent)
        self.fc_var = Lin(48,n_latent)
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self,x):
        n_bins = self.n_bins
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = x.view(-1,48)
        x = self.fc_mu(x)
        
        return x

class CNNDecoder(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=100,n_latent=10,n_hidden=50):
        super(CNNDecoder, self).__init__()

        self.n_latent = n_latent
        self.n_channels = n_channels

        self.n_bins = n_bins
        self.lin = Lin(n_latent,48)
        self.conv1 = ConvTranspose1d(in_channels=n_channels,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.constantpad1d1 = ConstantPad1d((1,0),0)
        self.conv2 = ConvTranspose1d(in_channels=n_channels*2,out_channels=n_channels,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = ConvTranspose1d(in_channels=n_channels,out_channels=2,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin2 = Lin(n_bins,n_bins)
        self.activation4 = Sigmoid()
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
    def forward(self,x):
        inp = x
        x = self.lin(inp)
        x = x.reshape(-1,4,12)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.constantpad1d1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.lin2(x)
        x = self.activation4(x) 

        return x

class MicroAutoEncoder(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=100,n_latent=10):
        super(MicroAutoEncoder, self).__init__()

        self.encoder = CNNEncoderVAE(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)
        self.decoder = CNNDecoder(n_channels=n_channels*2,n_bins=n_bins,n_latent=n_latent)

    def forward(self,x):

        bs = x.shape[0]
        
        latent = self.encoder(x)

        reconstruction = self.decoder(latent) 

        return reconstruction

def get_latent_var(model, dataloader, device, n_latent):
    dataset = dataloader.dataset
    latents = np.zeros((len(dataset), n_latent))

    jj=0
    for data in dataloader:
        bin0 = data
        bin0 = bin0.to(device)
        latent = model.encoder(bin0.float())
        bs = latent.shape[0]

        latents[jj:jj+bs,:]=latent.detach().cpu().numpy().reshape(bs, n_latent)
        jj+=bs
    
    return latents