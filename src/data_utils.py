from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.integrate import solve_ivp
from scipy.special import binom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def open_box_dataset():
    # Set path
    dpath = Path(__file__).parent.parent / "data" / "pysdm"

    # Train dataset
    ds_all = xr.open_dataset(dpath / "box64_train.nc", decode_timedelta=True)
    r_bins_edges = ds_all["mass_bin"]
    m_train = ds_all["dvdlnr"].sum(dim="mass_bin_idx")
    x_train = (
        (ds_all["dvdlnr"] / m_train).transpose("run", "time", "mass_bin_idx").to_numpy()
    )
    m_scale = m_train.max()
    m_train = (m_train / m_scale).to_numpy()
    n_bins = x_train.shape[2]
    dsd_time = (ds_all["time"] / np.timedelta64(1, "s")).to_numpy()

    # Test dataset
    ds_test = xr.open_dataset(dpath / "box64_test.nc", decode_timedelta=True)
    m_test = ds_test["dvdlnr"].sum(dim="mass_bin_idx")
    x_test = (
        (ds_test["dvdlnr"] / m_test).transpose("run", "time", "mass_bin_idx").to_numpy()
    )
    m_test = (m_test / m_scale).to_numpy()

    return (x_train, m_train, x_test, m_test, r_bins_edges, n_bins, dsd_time)


def open_erf_dataset(path=None, sample_time=None):
    if path is None:
        path = Path(__file__).parent.parent / "data"
        ds_all = xr.open_dataset(path / "congestus_coal_200m_train.nc")
        ds_test = xr.open_dataset(path / "congestus_coal_200m_test.nc")
    else:
        ds_all = xr.open_dataset(path + "_train.nc")
        ds_test = xr.open_dataset(path + "_test.nc")
    if sample_time is not None:
        ds_all = ds_all.isel(t=sample_time)
        ds_test = ds_test.isel(t=sample_time)
    r_bins_edges = ds_all["rbin_l"]
    m_train = ds_all["dmdlnr"].sum(dim="bin").transpose("loc", "t")
    x_train = (ds_all["dmdlnr"] / m_train).transpose("loc", "t", "bin").to_numpy()
    m_scale = m_train.max()
    m_train = (m_train / m_scale).to_numpy()
    n_bins = x_train.shape[2]
    dsd_time = ds_all["t"].to_numpy()
    dsd_time = dsd_time - dsd_time[0]
    m_test = ds_test["dmdlnr"].sum(dim="bin").transpose("loc", "t")
    x_test = (ds_test["dmdlnr"] / m_test).transpose("loc", "t", "bin").to_numpy()
    m_test = (m_test / m_scale).to_numpy()

    return (x_train, m_train, x_test, m_test, r_bins_edges, n_bins, dsd_time)


# define code for taking xarray Dataset and doing train-test split on it


def split_by_index(ds: xr.Dataset, dim: str, test_size: float, random_state: int = 0):
    """
    Splits a Dataset along one integer dimension into train/test.
    Returns (ds_train, idx_train, ds_test, idx_test).
    """
    idx = np.arange(ds.sizes[dim])
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    return ds.isel({dim: train_idx}), train_idx, ds.isel({dim: test_idx}), test_idx


def prepare(ds_sub: xr.Dataset, m_scale: float):
    """
    From a dataset returns (x, m) arrays:
        x[loc, t, bin] = normalized DSD across bins
        m[loc, t]      = mass fraction / m_scale
    """
    dmdlnr = ds_sub["dmdlnr"]
    # sum over bin → shape (t, loc); then transpose → (loc, t)
    m = dmdlnr.sum(dim="bin").transpose("loc", "t")
    # x has shape (loc, t, bin)
    x = (dmdlnr / m).transpose("loc", "t", "bin")
    return x.to_numpy(), (m / m_scale).to_numpy()


"""
Opens any dataset (e.g., RICO or Congestus) and does a specified train(-calibrate-)test split.
output depends on if a calibration set is specified or not
"""


def open_mass_dataset(
    name,
    data_dir,
    filepath=None,
    sample_time=None,
    test_size=0.2,
    calib_size=None,
    random_state=1952,
    m_scale=None,
):
    """
    Opens a *.nc named name under data_dir, splits into train/test(/calib),
    normalizes dmdlnr to get DSD and returns numpy arrays.
    """
    # 1) load
    if filepath is None:
        filepath = (data_dir / name).with_suffix(".nc")
    ds = xr.open_dataset(filepath)

    # 2) optional subsample in time
    if sample_time is not None:
        ds = ds.isel(t=sample_time)

    # 3) train/test split on 'loc'
    ds_train, idx_train, ds_test, idx_test = split_by_index(
        ds, dim="loc", test_size=test_size, random_state=random_state
    )

    # 4) calibration split from ds_train if requested
    ds_calib = None
    if calib_size is not None:
        # convert calib_size relative to the full dataset → relative to train only
        calib_size = calib_size / (1 - test_size)
        ds_train, idx_train, ds_calib, idx_calib = split_by_index(
            ds_train, dim="loc", test_size=calib_size, random_state=random_state
        )

    # 5) compute global m_scale from full ds_train (before calib‐split)
    if m_scale is None:
        ds_for_scale = (
            ds_train if ds_calib is None else xr.concat([ds_train, ds_calib], dim="loc")
        )
        m_scale = (
            ds_for_scale["dmdlnr"]
            .sum(dim="bin")
            .transpose("loc", "t")
            .max()
            .item()  # scalar
        )

    x_train, m_train = prepare(ds_train, m_scale)
    x_test, m_test = prepare(ds_test, m_scale)

    # gather outputs
    outputs = {
        "x_train": x_train,
        "m_train": m_train,
        "idx_train": idx_train,
        "x_test": x_test,
        "m_test": m_test,
        "idx_test": idx_test,
        "r_bins_edges": ds["rbin_l"].to_numpy(),
        "n_bins": x_train.shape[-1],
        "dsd_time": ds["t"].to_numpy() - ds["t"].to_numpy()[0],
        "m_scale": m_scale,
    }

    if ds_calib is not None:
        x_calib, m_calib = prepare(ds_calib, m_scale)
        outputs.update({"x_calib": x_calib, "m_calib": m_calib, "idx_calib": idx_calib})

    return outputs


def open_congestus_dataset(
    sample_time=None,
    test_size=0.2,
    calib_size=None,
    random_state=1952,
    data_dir=Path(__file__).parent.parent / "data",
):
    return open_mass_dataset(
        name="congestus_coal_200m",
        data_dir=data_dir,
        sample_time=sample_time,
        test_size=test_size,
        calib_size=calib_size,
        random_state=random_state,
    )


def open_rico_dataset(
    sample_time=None,
    test_size=0.2,
    calib_size=None,
    random_state=1952,
    data_dir=Path(__file__).parent.parent / "data",
):
    return open_mass_dataset(
        name="rico_coal_200m",
        data_dir=data_dir,
        sample_time=sample_time,
        test_size=test_size,
        calib_size=calib_size,
        random_state=random_state,
    )


# Train and calibrate on congestus, test on RICO
def open_congestus_calib_train_rico_test(
    calib_size=0.3,
    sample_time=None,
    random_state=1952,
    data_dir=Path(__file__).parent.parent / "data",
):
    """
    1) Loads congestus_coal_200m.nc and splits *all* of it into train/calib
       according to calib_size.
    2) Loads rico_coal_200m.nc in full as the sacred test set.
    3) Computes m_scale from the entire congestus dataset (train+calib).
    4) Returns dict of numpy arrays:
       x_train, m_train, x_calib, m_calib, x_test, m_test, r_bins, n_bins, dsd_time
    """
    # --- 1) load both datasets
    cong_path = (data_dir / "congestus_coal_200m").with_suffix(".nc")
    rico_path = (data_dir / "rico_coal_200m").with_suffix(".nc")

    ds_cong = xr.open_dataset(cong_path)
    ds_rico = xr.open_dataset(rico_path)

    # --- 2) optional time‐subsample
    if sample_time is not None:
        ds_cong = ds_cong.isel(t=sample_time)
        ds_rico = ds_rico.isel(t=sample_time)
    else:
        # default: keep all congestus timesteps, trim rico to match
        nt_cong = ds_cong.sizes["t"]
        ds_rico = ds_rico.isel(t=slice(0, nt_cong))

    # --- 3) split congestus into train/calib
    ds_train, idx_train, ds_calib, idx_calib = split_by_index(
        ds_cong, dim="loc", test_size=calib_size, random_state=random_state
    )

    # --- 4) test set is the *entire* rico dataset
    ds_test = ds_rico

    # --- 5) compute global m_scale from congestus only
    m_scale = ds_cong["dmdlnr"].sum(dim="bin").transpose("loc", "t").max().item()

    # --- 6) prepare arrays
    x_train, m_train = prepare(ds_train, m_scale)
    x_calib, m_calib = prepare(ds_calib, m_scale)
    x_test, m_test = prepare(ds_test, m_scale)

    # --- 7) other metadata
    r_bins = ds_cong["rbin_l"].to_numpy()
    n_bins = x_train.shape[-1]
    dsd_time = ds_cong["t"].to_numpy() - ds_cong["t"].to_numpy()[0]

    return {
        "x_train": x_train,
        "m_train": m_train,
        "idx_train": idx_train,
        "x_calib": x_calib,
        "m_calib": m_calib,
        "idx_calib": idx_calib,
        "x_test": x_test,
        "m_test": m_test,
        "idx_test": idx_test,
        "r_bins_edges": r_bins,
        "n_bins": n_bins,
        "dsd_time": dsd_time,
    }


# Train on congestus, calibrate and test on RICO
def open_congestus_train_rico_calib_test(
    test_size=0.3,
    sample_time=None,
    random_state=1952,
    data_dir=Path(__file__).parent.parent / "data",
):
    """
    1) Loads congestus_coal_200m.nc as the *entire* training set.
    2) Loads rico_coal_200m.nc and splits it into calib/test by test_size.
    3) Computes m_scale from (congestus + rico_calib).
    4) Returns dict of numpy arrays:
       x_train, m_train, x_calib, m_calib, x_test, m_test, r_bins, n_bins, dsd_time
    """
    cong_path = (data_dir / "congestus_coal_200m").with_suffix(".nc")
    rico_path = (data_dir / "rico_coal_200m").with_suffix(".nc")

    ds_cong = xr.open_dataset(cong_path)
    ds_rico = xr.open_dataset(rico_path)

    if sample_time is not None:
        ds_cong = ds_cong.isel(t=sample_time)
        ds_rico = ds_rico.isel(t=sample_time)
    else:
        nt_cong = ds_cong.sizes["t"]
        ds_rico = ds_rico.isel(t=slice(0, nt_cong))

    # train = all congestus
    ds_train = ds_cong

    # split rico into calib / test
    rico_calib, idx_calib, rico_test, idx_test = split_by_index(
        ds_rico, dim="loc", test_size=test_size, random_state=random_state
    )
    ds_calib = rico_calib
    ds_test = rico_test

    # m_scale from everything except the sacred test set
    ds_for_scale = xr.concat([ds_train, ds_calib], dim="loc")
    m_scale = ds_for_scale["dmdlnr"].sum(dim="bin").transpose("loc", "t").max().item()

    x_train, m_train = prepare(ds_train, m_scale)
    x_calib, m_calib = prepare(ds_calib, m_scale)
    x_test, m_test = prepare(ds_test, m_scale)

    r_bins = ds_cong["rbin_l"].to_numpy()
    n_bins = x_train.shape[-1]
    dsd_time = ds_cong["t"].to_numpy() - ds_cong["t"].to_numpy()[0]

    return {
        "x_train": x_train,
        "m_train": m_train,
        "idx_train": idx_train,
        "x_calib": x_calib,
        "m_calib": m_calib,
        "idx_calib": idx_calib,
        "x_test": x_test,
        "m_test": m_test,
        "idx_test": idx_test,
        "r_bins_edges": r_bins,
        "n_bins": n_bins,
        "dsd_time": dsd_time,
    }


def open_congestus_train_rico_test(
    sample_time=None, random_state=1952, data_dir=Path(__file__).parent.parent / "data"
):
    """
    1) Load all congestus_coal_200m.nc (61 timesteps).
    2) Load all rico_coal_200m.nc (101 timesteps).
    3) If sample_time is provided, apply to both. Otherwise trim RICO
       to the first nt_cong timesteps (default nt_cong=61).
    4) Compute m_scale from congestus only.
    5) Prepare and return numpy arrays:
       x_train, m_train, x_test, m_test, r_bins_edges, n_bins, dsd_time
    """
    cong_path = (data_dir / "congestus_coal_200m").with_suffix(".nc")
    rico_path = (data_dir / "rico_coal_200m").with_suffix(".nc")

    ds_cong = xr.open_dataset(cong_path)
    ds_rico = xr.open_dataset(rico_path)

    # --- time alignment ---
    if sample_time is not None:
        ds_cong = ds_cong.isel(t=sample_time)
        ds_rico = ds_rico.isel(t=sample_time)
    else:
        # trim rico to first nt_cong timesteps
        nt_cong = ds_cong.sizes["t"]
        ds_rico = ds_rico.isel(t=slice(0, nt_cong))

    # --- define train/test sets ---
    ds_train = ds_cong
    ds_test = ds_rico

    # --- compute normalization scale from congestus only ---
    m_scale = ds_train["dmdlnr"].sum(dim="bin").transpose("loc", "t").max().item()

    # --- prepare numpy arrays ---
    x_train, m_train = prepare(ds_train, m_scale)
    x_test, m_test = prepare(ds_test, m_scale)

    # --- metadata ---
    r_bins_edges = ds_cong["rbin_l"].to_numpy()
    n_bins = x_train.shape[-1]
    # relative time axis (seconds or minutes as in the file)
    dsd_time = ds_cong["t"].to_numpy() - ds_cong["t"].to_numpy()[0]

    return {
        "x_train": x_train,  # shape: (n_train_loc, nt, n_bins)
        "m_train": m_train,  # shape: (n_train_loc, nt)
        "idx_train": idx_train,
        "x_test": x_test,  # shape: (n_test_loc,  nt, n_bins)
        "m_test": m_test,  # shape: (n_test_loc,  nt)
        "idx_test": idx_test,
        "r_bins_edges": r_bins_edges,  # 1D array, length = n_bins+1 or n_bins
        "n_bins": n_bins,
        "dsd_time": dsd_time,  # 1D array, length = nt
    }


"""
The following function generates indices for the bootstrap replicate of a provided training dataset.
In other words, it resamples the sample indices with replacement.
"""


def resampled_indices(test_data):
    return np.random.randint(0, len(test_data), size=len(test_data))


# Create torch dataset
class NormedBinDatasetDzDt(Dataset):
    def __init__(self, dmdlnr_normed, dsd_time, M):
        self.nbin = dmdlnr_normed.shape[2]
        self.t = dsd_time
        self.dt = self.t[1] - self.t[0]
        self.x = dmdlnr_normed.reshape(-1, 1, self.nbin).astype(np.float32)
        self.dx = np.gradient(dmdlnr_normed, axis=1).reshape(-1, 1, self.nbin).astype(
            np.float32
        ) / self.dt.astype(np.float32)
        self.M = M.reshape(-1, 1, 1).astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx, :], self.dx[idx, :], self.M[idx]


# Create torch dataset
class NormedBinDatasetAR(Dataset):
    def __init__(self, dmdlnr_normed, M, lag=1):
        self.nbin = dmdlnr_normed.shape[2]
        self.lag = lag
        self.bin0 = (
            []
        )  # dmdlnr_normed.astype(np.float32)[:,:-1*lag,:].reshape([-1, 1, self.nbin])
        self.bin1 = (
            []
        )  # dmdlnr_normed.astype(np.float32)[:,lag:,:].reshape([-1, 1, self.nbin])
        self.M = []  # M.astype(np.float32).reshape([-1, 1, 1])

        for i in range(dmdlnr_normed.shape[1] - lag):
            self.bin0.append(dmdlnr_normed[:, i : i + lag, :].astype(np.float32))
            self.bin1.append(dmdlnr_normed[:, i + lag, :].astype(np.float32))
            self.M.append(M[:, i + lag].astype(np.float32))

        self.bin0 = np.array(self.bin0).reshape([-1, lag, self.nbin])
        self.bin1 = np.array(self.bin1).reshape([-1, 1, self.nbin])
        self.M = np.array(self.M).reshape([-1, 1, 1])

    def __len__(self):
        return int(self.bin0.shape[0])

    def __getitem__(self, idx):
        return self.bin0[idx, :], self.bin1[idx, :], self.M[idx]


def sindy_library_tensor(z, latent_dim, poly_order):
    library_dim = library_size(latent_dim, poly_order)
    if len(z.shape) == 1:
        z = z.unsqueeze(0)
    if len(z.shape) == 2:
        z = z.unsqueeze(1)
    new_library = torch.zeros(z.shape[0], z.shape[1], library_dim)

    # i = 0: constant
    idx = 0
    new_library[:, :, idx] = 1.0

    idx += 1
    # i = 1:nl + 1 -> first order
    if poly_order >= 1:
        new_library[:, :, idx : idx + latent_dim] = z

    idx += latent_dim
    # second order
    if poly_order >= 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                new_library[:, :, idx] = z[:, :, i] * z[:, :, j]
                idx += 1

    # third order+
    for order in range(3, poly_order + 1):
        for idxs in combinations_with_replacement(range(latent_dim), order):
            term = z[:, :, idxs[0]]
            for i in idxs[1:]:
                term = term * z[:, :, i]
            new_library[:, :, idx] = term
            idx += 1

    return new_library


def library_size(n, poly_order):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    return l


def simulate(z0, T, dz_network, z_lim):
    def f(t, z):
        n_latent = z.size
        dz = dz_network(torch.Tensor(z)).squeeze().detach().numpy()
        for il in range(n_latent):
            if (z[il] >= z_lim[il][1]) or (z[il] <= z_lim[il][0]):
                dz[il] = 0.0
        return dz

    sol = solve_ivp(f, [T[0], T[-1]], z0, method="RK45", t_eval=T)
    Z = sol.y.T
    return Z


def champion_calculate_weights(ds, lambda1_metaweight=0.5, lambda3=1.0):
    """
    See Champion et al. supplementary materials for information.

    :param ds: Training dataset
    :param lambda1_metaweight: lambda1 is specified as "slightly less than", this sets that
    :param lambda3: Reconstruction weight, currently just set to 1.0 but this function allows this to be
                    programmatically changed.
    :return: lambda1, lambda2, lambda3
    """
    xx = np.squeeze(ds.x)
    dx = np.squeeze(ds.dx)
    xxl2 = np.linalg.norm(xx, ord=2, axis=1) ** 2
    dxl2 = np.linalg.norm(dx, ord=2, axis=1) ** 2
    lambda1 = xxl2.sum() / dxl2.sum() * lambda1_metaweight
    lambda2 = lambda1 / 1e2  # 2 orders of magnitude smaller
    return lambda1, lambda2, lambda3
