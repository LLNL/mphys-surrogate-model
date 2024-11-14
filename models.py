import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import Linear as Lin, ReLU, Sigmoid, ConstantPad1d
import pysindy as ps

class BinDataset2(Dataset):
    def __init__(self, data):
        self.bin0 = data

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :]
    
class BinDataset(Dataset):
    def __init__(self, data):
        self.bin0 = data

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :, :]

def normalize_data(bin0):
    # data QC
    N_threshold = [0.1 * 1e6,  1e13] #0.1 - 10,000 / cm^3
    M_threshold = (1e-9, 1e16) #0.01 - 10 g / m^3
    casemask = np.zeros(bin0.shape[0])
    case_idx = np.arange(0, bin0.shape[0], 1)
    binsums = np.sum(bin0, axis=2)
    for i in range(0, bin0.shape[0]):
        if np.any(bin0[i, 0, :] > N_threshold[1]) or np.any(bin0[i, 1, :] > M_threshold[1]):
            casemask[i] = False
        elif binsums[i, 0] < N_threshold[0] or binsums[i, 1] < M_threshold[0]:
            casemask[i] = False
        else:
            casemask[i] = True
    bin0 = bin0[casemask == 1.0, :]
    case_idx = case_idx[casemask == 1.0]

    # Normalization?
    binsums = np.sum(bin0, axis=2)
    momscales = np.max(binsums, axis=0)
    bin0[:, 0, :] /= momscales[0]
    bin0[:, 1, :] /= momscales[1]

    return (bin0, case_idx)

def create_timeseries_dataloader(ds):
    bs = ds['time_save_spec'].size
    (data_loader, _, _, t_idx) = create_dataloader(None, bs, tvt_split=(100, 0, 0), shuffle=False, ds=ds, return_idx=True)
    return (data_loader, t_idx)

def create_dataloader(filepath, bs, tvt_split = (50, 25, 25), shuffle=True, ds=None, return_idx=False):
    if filepath is not None:
        ds = xr.open_mfdataset(filepath + "*.nc", combine='nested', concat_dim='run')
        r_bins_edges = np.logspace(np.log10(0.1 * 1e-6), np.log10(10 * 1e-3), 101, endpoint=True,)

        # flatten the data
        data = ds.stack(case=("run","time_save_spec","height"))
    else:
        assert ds is not None
        data = ds.stack(case=("time_save_spec",))

    nc = data['case'].size
    nb = data['wet_spectrum_bin_index'].size

    bin0 = np.zeros((nc, 2, nb))
    bin0[:, 0, :] = data['wet spectrum'].to_numpy().T
    bin0[:, 1, :] = data['dvdlnr'].to_numpy().T

    # mean distributions
    (bin0, case_idx) = normalize_data(bin0)

    ncase, _, nb = bin0.shape
    print(f"Found {ncase} samples out of {nc}")

    # Test-train split
    assert sum(tvt_split) == 100
    assert tvt_split[0] > 0
    idx = np.arange(0,bin0.shape[0])
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(idx)

    # Train
    trainidx = idx[0:int(tvt_split[0]/100 * bin0.shape[0])]
    bin0_train = bin0[trainidx,:]
    traindataset = BinDataset(bin0_train)
    train_dataloader = DataLoader(traindataset, batch_size=bs)

    # Validate
    if tvt_split[1] > 0:
        validx = idx[int(tvt_split[0]/100 * bin0.shape[0]):int(sum(tvt_split[0:1])/100 * bin0.shape[0])]
        bin0_val = bin0[validx,:]
        valdataset = BinDataset(bin0_val)
        val_dataloader = DataLoader(valdataset, batch_size=bs)
    else:
        val_dataloader = None
    
    # Testing
    if tvt_split[2] > 0:
        testidx = idx[int(sum(tvt_split[0:1])/100 * bin0.shape[0])]
        bin0_test = bin0[testidx,:]
        testdataset = BinDataset(bin0_test)
        test_dataloader = DataLoader(testdataset, batch_size=bs)
    else:
        test_dataloader = None
    
    print("train ",int(tvt_split[0]/100 * bin0.shape[0]),"val ",int(tvt_split[1]/100 * bin0.shape[0]),"test ",int(tvt_split[2]/100 * bin0.shape[0]))
    if return_idx:
        return (train_dataloader, test_dataloader, val_dataloader, case_idx)
    else:
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
        self.conv1 = Conv1d(in_channels=n_channels,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.conv2 = Conv1d(in_channels=n_channels*2,out_channels=n_channels*4,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = Conv1d(in_channels=n_channels*4,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin1 = Lin(48,n_latent)

        self.layer_id = ["conv1", "conv2", "conv3", "lin1"]
        self.layers = [self.conv1, self.conv2, self.conv3, self.lin1]
        
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
        x = self.lin1(x)
        
        return x
    
    def get_weights(self):
        weights = {}
        biases = {}
        for (i, layer) in enumerate(self.layers):
            weights[self.layer_id[i]] = layer.weight
            biases[self.layer_id[i]] = layer.bias

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[self.layer_id[i]]
            layer.bias.data = biases[self.layer_id[i]]

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

        self.layer_id = ["lin", "conv1", "conv2", "conv3", "lin2"]
        self.layers = [self.lin, self.conv1, self.conv2, self.conv3, self.lin2]
        
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
    
    def get_weights(self):
        weights = {}
        biases = {}
        for (i, layer) in enumerate(self.layers):
            weights[self.layer_id[i]] = layer.weight
            biases[self.layer_id[i]] = layer.bias

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[self.layer_id[i]]
            layer.bias.data = biases[self.layer_id[i]]

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


