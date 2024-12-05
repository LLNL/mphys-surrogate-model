from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import torch
import random
from scipy.special import binom


# Utilities for training CNN on 1-channel and 2-channel data from 1d KiD runs
class BinDataset1C(Dataset):
    def __init__(self, data):
        self.bin0 = data

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :]
    
class BinDataset2C(Dataset):
    def __init__(self, data):
        self.bin0 = data

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :, :]
    
def normalize_data_1d(bin0):
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
    (bin0, case_idx) = normalize_data_1d(bin0)

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
    traindataset = BinDataset2C(bin0_train)
    train_dataloader = DataLoader(traindataset, batch_size=bs)

    # Validate
    if tvt_split[1] > 0:
        validx = idx[int(tvt_split[0]/100 * bin0.shape[0]):int(sum(tvt_split[0:1])/100 * bin0.shape[0])]
        bin0_val = bin0[validx,:]
        valdataset = BinDataset2C(bin0_val)
        val_dataloader = DataLoader(valdataset, batch_size=bs)
    else:
        val_dataloader = None
    
    # Testing
    if tvt_split[2] > 0:
        testidx = idx[int(sum(tvt_split[0:1])/100 * bin0.shape[0])]
        bin0_test = bin0[testidx,:]
        testdataset = BinDataset2C(bin0_test)
        test_dataloader = DataLoader(testdataset, batch_size=bs)
    else:
        test_dataloader = None
    
    print("train ",int(tvt_split[0]/100 * bin0.shape[0]),"val ",int(tvt_split[1]/100 * bin0.shape[0]),"test ",int(tvt_split[2]/100 * bin0.shape[0]))
    if return_idx:
        return (train_dataloader, test_dataloader, val_dataloader, case_idx)
    else:
        return (train_dataloader, test_dataloader, val_dataloader)

# Utilities for end-to-end training of box model
class E2EDataset(Dataset):
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return (self.x[idx, :, :], self.dx[idx, :, :])

def create_e2e_dataloader(ds, cnn=False, shuffle_runs=True, normx = True, normdx = True, batch_size=100, tvt_split = (80, 10, 10), ):
    one_sec = np.timedelta64(1, 's')
    t = (ds['time'] / one_sec).to_numpy()
    dt = int((ds['time'].isel(time=1) - ds['time'].isel(time=0)) / one_sec)
    x = ds['dvdlnr'].transpose('run','time','mass_bin_idx').to_numpy()

    dx = np.gradient(x, axis=1) / dt

    if normx:
        x_norm = np.max(x)
    else:
        x_norm = 1.0
    
    if normdx:
        dx_norm = np.max(dx)
        t_norm = x_norm / dx_norm
    else:
        t_norm = 1.0

    x = x / x_norm
    dx = dx / x_norm * t_norm
    t = t / t_norm

    x_data = x.copy()
    dx_data = dx.copy()

    if shuffle_runs:
        shuffle_idx = ds['run'].data
        random.shuffle(shuffle_idx)
        x = x[shuffle_idx, :, :]
        dx = dx[shuffle_idx, :, :]

    if cnn:
        old_shape = x.shape
        print(f"{old_shape[0]} runs with {old_shape[1]} timesteps each")
        x.shape = (old_shape[0] * old_shape[1], 1, old_shape[2])
        dx.shape = x.shape

    # Train
    x_train = x[0:int(tvt_split[0]/100 * x.shape[0])]
    dx_train = dx[0:int(tvt_split[0]/100 * x.shape[0])]
    traindataset = E2EDataset(x_train, dx_train)
    train_dataloader = DataLoader(traindataset, batch_size=batch_size)

    # Validate
    if tvt_split[1] > 0:
        x_val = x[int(tvt_split[0]/100 * x.shape[0]):int(sum(tvt_split[0:2])/100 * x.shape[0])]
        dx_val = x[int(tvt_split[0]/100 * x.shape[0]):int(sum(tvt_split[0:2])/100 * x.shape[0])]
        valdataset = E2EDataset(x_val, dx_val)
        val_dataloader = DataLoader(valdataset, batch_size=batch_size)
    else:
        val_dataloader = None
    
    # Testing
    if tvt_split[2] > 0:
        x_test = x[int(sum(tvt_split[0:2])/100 * x.shape[0]):]
        dx_test = dx[int(sum(tvt_split[0:2])/100 * x.shape[0]):]
        testdataset = E2EDataset(x_test, dx_test)
        test_dataloader = DataLoader(testdataset, batch_size=batch_size)
    else:
        test_dataloader = None

    data = (x_data, dx_data, t)
    norms = (x_norm, t_norm)
    data_loaders = (train_dataloader, val_dataloader, test_dataloader)

    return (data, norms, data_loaders)

def sindy_library_tensor(z, latent_dim, poly_order):
    # not implemented for order 2 and higher terms
    library_dim = library_size(latent_dim, poly_order)
    new_library = torch.zeros(z.shape[0], z.shape[1], library_dim)

    idx = 0
    # i = 0: constant
    new_library[:, :, idx] = 1.0

    idx += 1
    # i = 1:nl + 1 -> first order
    new_library[:, :, idx:idx + latent_dim] = z

    idx += latent_dim
    # second order
    if poly_order >= 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                new_library[:, :, idx] = z[:, :, i] * z[:, :, j]
                idx += 1

    return new_library

def library_size(n, poly_order):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    return l