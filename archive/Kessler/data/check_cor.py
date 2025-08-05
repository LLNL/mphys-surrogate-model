""" 
This script checks spatial correlations between each gridbox and the others. These are saved. 
"""

import numpy as np

# input name of data file (all but the .nc!)
data_name = input()
dset = np.load(data_name + "_processed.npy")
coords = np.load(data_name + "_coords_filtered.npy")
var_means = dset[:, :, 1:].mean(axis=(0, 1))
dset[:, :, 1:] /= var_means[None, None, :]  # normalize each variable by its own mean

"""
1. Compute nearest-neighbor distances.
-We want to pick the minimal radius such that every gridbox has AT LEAST one neighbor
"""

from scipy.spatial import KDTree

tree = KDTree(coords)

# query the 2 nearest neighbors:
#  - the 1st will be the point itself at distance=0
#  - the 2nd is the true nearest other point
dists, inds = tree.query(coords, k=2)
# nearest‐neighbor distance for each grid‐box:
nn_dist = dists[:, 1]
# minimal r so that EVERY box has at least one neighbor:
r_min = np.ceil(nn_dist.max())  # take the ceiling just to add a fudge factor

print("Done with computing nearest-neighbor distances.")

"""
2. Save "close" gridboxes for each gridbox. 
-These are gridboxes within r_min distance.
"""

# for each point, get all grid indices within distance r_min
ball_lists = tree.query_ball_point(coords, r=r_min)
neighbor_list = [
    [j for j in nbrs if j != i] for i, nbrs in enumerate(ball_lists)
]  # store in neighbor_list

import pickle

# saving list of neighbors for each point
with open(data_name + "_neighbors.pkl", "wb") as f:
    pickle.dump(neighbor_list, f)

print("Done with computing neighboring gridboxes.")

"""
3. Pre-process data 
-To properly check for correlations, since neighboring gridboxes could have similar initial conditions (and hence similar time series),
we need to check that, even if we detrend each time series separately, the residuals are uncorrelated.
-We fit a simple AR(p) model to each series and thereby extract residuals
"""

from statsmodels.tsa.ar_model import AutoReg

ar_order = 1


def prewhiten(ts, ar_order=ar_order):
    # 1) Fit AR(p)
    model = AutoReg(ts, lags=ar_order, old_names=False).fit()
    # 2) Get residuals
    return model.resid


p = 1  # order for autoregressive model
dset_residuals = np.empty(
    (dset.shape[0], dset.shape[1] - p, dset.shape[2]), dtype=float
)  # get residuals for model

# we will just replace every time series in dset, as we don't need the original data from hereonafter
for i, ts in enumerate(dset):
    tmp = np.array(
        [prewhiten(ts[:, j + 1], p) for j in range(len(ts[0]) - 1)]
    ).T  # save residuals
    dset_residuals[i] = np.hstack(
        (ts[p:, 0:1], tmp)
    )  # concatenate residuals with times
del dset  # delete original dataset as we don't need it anymore

print("Done with pre-processing data.")

"""
4. Run correlation analysis between residuals
-Uses CCF to run Li-McLeod/Ljung-Box cross-portmanteau test
-Also uses Pillai's trace to run a test of significance for a lienar correlation test
"""

# use FFT to efficiently compute the CCFs between two time series
from numpy.fft import fft, ifft


def multivar_ccf_fft_squared(
    X, Y, maxlags=None, unbiased=False, demean=True, standardize=True
):
    """
    Compute cross‐correlation functions between columns of X and Y via FFT, then take the RMS to compute the overall "energy" of the correlation.

    Parameters
    ----------
    X : array_like, shape (T, p)
    Y : array_like, shape (T, q)
    maxlags : int, optional
        Maximum lag (in both directions). If None, uses full range T-1.
    unbiased : bool, default False
        If True, uses 1/(T-|lag|) normalization; else 1/T (biased).
    demean : bool, default True
        Subtract column‐means before correlation.
    standardize : bool, default True
        Divide by (σ_xᵢ·σ_yⱼ) so that ccf(0) == 1 on the diagonal.

    Returns
    -------
    lags : ndarray, shape (2*L+1,)
        The lag vector from -L…+L.
    ccf  : ndarray, shape (2*L+1, p, q)
        ccf[k, i, j] is correlation between X[:,i] and Y[:,j] at lag lags[k].
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    T, p = X.shape
    T2, q = Y.shape
    if T2 != T:
        raise ValueError("X and Y must have same length in time axis")

    # 1) Demean (and optionally standardize)
    if demean:
        x = X - X.mean(axis=0)
        y = Y - Y.mean(axis=0)
    else:
        x, y = X, Y

    if standardize:
        sx = x.std(axis=0, ddof=0)
        sy = y.std(axis=0, ddof=0)
        # avoid division-by-zero
        sx[sx == 0] = 1.0
        sy[sy == 0] = 1.0
    else:
        sx = np.ones(p)
        sy = np.ones(q)

    # 2) Choose FFT length
    full_lags = T - 1
    L = full_lags if maxlags is None else int(maxlags)
    if L > full_lags:
        raise ValueError("maxlags must be <= T-1")
    nfft = 1 << int(np.ceil(np.log2(2 * T - 1)))  # next power of 2 ≥ 2T-1

    # 3) FFT along time‐axis
    Xf = fft(x, nfft, axis=0)  # shape (nfft, p)
    Yf = fft(y, nfft, axis=0)  # shape (nfft, q)

    # 4) Build cross‐spectrum via broadcasting
    #    shape (nfft, p, q)
    cross_spec = Xf.conj()[:, :, None] * Yf[:, None, :]

    # 5) Inverse FFT → circular cross-correlation
    cc_circular = nfft * ifft(cross_spec, axis=0).real

    # 6) Extract lags from -(T-1)…+(T-1)
    #    In FFT output, lag zero is at index 0,
    #    negative lags are at indices nfft-(1…T-1).
    idx = np.concatenate(
        (np.arange(nfft - full_lags, nfft), np.arange(0, full_lags + 1))
    )
    cc_full = cc_circular[idx, :, :]  # shape = (2T-1, p, q)

    # 7) Trim to desired maxlags
    center = full_lags
    start = center - L
    stop = center + L + 1
    ccf_num = cc_full[start:stop, :, :]  # shape = (2L+1, p, q)

    # 8) Normalization
    if unbiased:
        norm_factor = (T - np.abs(np.arange(-L, L + 1)))[:, None, None]
    else:
        norm_factor = T
    # divide by (σ_x * σ_y)
    denom = (sx[:, None] * sy[None, :])[None, :, :]  # shape = (1,p,q)
    ccf = ccf_num / norm_factor / denom

    # combine into unified measure; not across lags, but across spatial dimensions
    # Frobenius norm squared
    return np.sum(ccf**2, axis=(1, 2))


from scipy.stats import chi2


def cross_lb_fft_test(
    X, Y, m=None, ccf_fft=multivar_ccf_fft_squared, tol=None, ar_order=ar_order
):
    """
    Cross-portmanteau test via FFT-CCF + whitening.

    Parameters
    ----------
    X : array (n, p)
    Y : array (n, q)
    m : int
        Max lag. Default is full range n-1.
    ccf_fft : call-able
        Your FFT-based CCF-squared routine returning a (2m+1,) array of
        Frobenius-norm-squared cross-covariances at lags -m..+m.
    tol : tolerance for Cholesky decomposition

    Returns
    -------
    Q       : float
    df      : int  (p*q*m)
    p_value : float
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    n2, q = Y.shape
    if n2 != n:
        raise ValueError("X and Y must have same length")

    # 1) demean
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)

    # 2) sample covariances
    Sx = (X0.T @ X0) / (n - 1)
    Sy = (Y0.T @ Y0) / (n - 1)
    cons_flag_X = cons_flag_Y = False
    # Check if covariances indicate a near constant time series. If so, add flag.
    if np.linalg.norm(Sx) < tol:
        cons_flag_X = not cons_flag_X
    if np.linalg.norm(Sy) < tol:
        cons_flag_Y = not cons_flag_Y

    # 3) Choose maximum lag. By default, this will be n-1.
    full_lags = n - 1
    L = full_lags if m is None else int(m)
    if L > full_lags:
        raise ValueError("m must be <= n-1")
    m = L

    # 4) call FFT‐CCF: no further demeaning, no standardization, biased norm
    #    returns 2*m+1 vector for lags -m..+m
    ccf2 = ccf_fft(X0, Y0, maxlags=m, unbiased=True, demean=False, standardize=False)
    # Sanity check
    if ccf2.shape[0] != 2 * m + 1:
        raise ValueError("ccf_fft must return 2*m+1 array")

    # 5) extract positive lags 1..m
    #    index zero   = lag -m
    #    index m      = lag  0
    #    index m+k    = lag +k
    pos = ccf2[m + 1 : m + 1 + m]  # length = m

    # 6) build Q-statistic
    ks = np.arange(1, m + 1)
    Q = n * (n + 2) * np.sum(pos / (n - ks))

    # 7) degrees of freedom and p‐value
    npar = 2 * ar_order  # assuming detrending using AR(ar_order)
    df = p * q * m - npar
    pval = chi2.sf(Q, df)
    return cons_flag_X, cons_flag_Y, Q, df, pval


from statsmodels.api import add_constant
from scipy import stats

tol = 1e-8  # tolerance for adding white noise (in case of near-constant time series)
rng = np.random.default_rng(1952)  # instantiate noise generator

CCFs_pvals = neighbor_list  # preallocate p-values for CCF test
n_vars = dset_residuals.shape[2] - 1
linear_pvals = (
    neighbor_list  # preallocate p-values for linear fit between neighbors and gridbox
)
for i, ts_i in enumerate(
    dset_residuals[:, :, 1:]
):  # for each gridbox i and time series ts_i...
    ts_neighbors = dset_residuals[
        neighbor_list[i], :, 1:
    ]  # get all time series for neighbors
    # compute CCF values
    for j, ts_j in enumerate(ts_neighbors):
        # compute Li–McLeod/Ljung–Box cross-portmanteau test
        # for maximum lags, m, we use the heuristic that m ~ n ^ (1/3)
        CCFs_pvals[i][j] = list(
            (
                cross_lb_fft_test(
                    ts_i, ts_j, m=int(round(len(ts_i) ** (1 / 3))), tol=tol
                )[k]
                for k in (0, 1, -1)
            )
        )
        print(CCFs_pvals[i][j])

        # fit and score linear fit of gridbox j to predict gridbox i
        # add a bit of jitter (white noise) in case if it's near-constant
        tol_i = tol * np.maximum(np.std(ts_i, axis=0, keepdims=True), 1)
        ts_i += rng.normal(loc=0.0, scale=tol_i, size=ts_i.shape)
        tol_j = tol * np.maximum(np.std(ts_j, axis=0, keepdims=True), 1)
        ts_j += rng.normal(loc=0.0, scale=tol_j, size=ts_j.shape)
        exog = add_constant(ts_j)
        endog = ts_i

        # compute Pillai's trace and p-value for linear correlation test
        n = len(endog)
        J = np.eye(n) - np.ones((n, n)) / n  # build projection off intercept
        B = np.linalg.pinv(exog) @ endog  # fit coefficients
        Xj = exog[:, 1:]  # partition exog into [const | Xj]
        # H = B_j' Xj' J Xj B_j
        Bj = B[1:, :]  # shape (q×p)
        H = Bj.T @ (Xj.T @ J @ Xj) @ Bj
        # E = residual sum of squares×cross-prod
        E = (endog - exog @ B).T @ (endog - exog @ B)
        # canonical roots
        eigvals = np.linalg.eigvals(np.linalg.inv(E) @ H)
        V = np.sum(eigvals / (1 + eigvals)).real
        # F approx
        num = (2 * n - 2 * n_vars - 1) * V
        den = (n_vars**2) * (1 - V)
        df1 = n_vars**2
        df2 = 2 * n - 2 * n_vars - 1
        F = num / den
        linear_pvals[i][j] = 1 - stats.f.cdf(F, df1, df2)

# final arrays are of size (gridbox, neighbor gridboxes)
T = dset_residuals.shape[1]
# save portmanteau p-values as pkl file
with open(data_name + "_CCF_test.pkl", "wb") as f:
    pickle.dump(CCFs_pvals, f)
# save linear correlation test values as pkl file (same size as above)
with open(data_name + "_linear_test.pkl", "wb") as f:
    pickle.dump(linear_pvals, f)

print("Done with cross-correlation analysis.")
