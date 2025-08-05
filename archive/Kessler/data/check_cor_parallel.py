import os
import argparse
import pickle
import numpy as np
from scipy.spatial import KDTree
from statsmodels.tsa.ar_model import AutoReg
from numpy.fft import fft, ifft
from scipy.stats import chi2
from statsmodels.api import add_constant
import scipy.stats as stats
from multiprocessing import Pool, cpu_count


# ─────────────────────────────────────────────────────────────────────────────
# 1) FFT‐based Frobenius‐norm‐squared CCF (unchanged)
def multivar_ccf_fft_squared(
    X, Y, maxlags=None, unbiased=False, demean=True, standardize=True
):
    X = np.asarray(X)
    Y = np.asarray(Y)
    T, p = X.shape
    T2, q = Y.shape
    if T2 != T:
        raise ValueError("X and Y must share time axis length")
    # 1) demean
    if demean:
        x = X - X.mean(axis=0)
        y = Y - Y.mean(axis=0)
    else:
        x, y = X, Y
    # 2) standardize
    if standardize:
        sx = x.std(axis=0, ddof=0)
        sx[sx == 0] = 1.0
        sy = y.std(axis=0, ddof=0)
        sy[sy == 0] = 1.0
    else:
        sx = np.ones(p)
        sy = np.ones(q)
    # 3) FFT length
    full_lags = T - 1
    L = full_lags if maxlags is None else int(maxlags)
    if L > full_lags:
        raise ValueError("maxlags must be <= T-1")
    nfft = 1 << int(np.ceil(np.log2(2 * T - 1)))
    # 4) FFT & cross‐spectrum
    Xf = fft(x, nfft, axis=0)
    Yf = fft(y, nfft, axis=0)
    cross_spec = Xf.conj()[:, :, None] * Yf[:, None, :]
    # 5) circular cross‐corr
    cc_circ = nfft * ifft(cross_spec, axis=0).real
    # 6) re‐index lags
    idx = np.r_[np.arange(nfft - full_lags, nfft), np.arange(0, full_lags + 1)]
    cc_full = cc_circ[idx, :, :]
    # 7) trim
    center = full_lags
    ccf_num = cc_full[center - L : center + L + 1, :, :]
    # 8) normalize
    if unbiased:
        norm = (T - np.abs(np.arange(-L, L + 1)))[:, None, None]
    else:
        norm = T
    denom = (sx[:, None] * sy[None, :])[None, :, :]
    ccf = ccf_num / norm / denom
    return np.sum(ccf**2, axis=(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Cross‐portmanteau via FFT‐CCF + Ljung–Box
def cross_lb_fft_test(
    X, Y, m=None, ccf_fft=multivar_ccf_fft_squared, tol=None, ar_order=None
):
    """
    Returns (cons_flag_X, cons_flag_Y, Q, df, p_value).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    n2, q = Y.shape
    if n2 != n:
        raise ValueError("X and Y must share time length")

    # 1) demean
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    # 2) sample covariances & constant‐flag
    Sx = (X0.T @ X0) / (n - 1)
    Sy = (Y0.T @ Y0) / (n - 1)
    cons_flag_X = np.linalg.norm(Sx) < tol
    cons_flag_Y = np.linalg.norm(Sy) < tol

    # 3) choose lag‐window
    full_lags = n - 1
    L = full_lags if m is None else int(m)
    if L > full_lags:
        raise ValueError("m must be <= n-1")

    # 4) compute biased, demeaned, unstd CCF
    ccf2 = ccf_fft(X0, Y0, maxlags=L, unbiased=True, demean=False, standardize=False)
    if ccf2.shape[0] != 2 * L + 1:
        raise ValueError("ccf_fft must return 2*m+1 array")

    # 5) extract positive lags
    pos = ccf2[L + 1 : L + 1 + L]  # length = L
    ks = np.arange(1, L + 1)

    # 6) Q‐statistic
    Q = n * (n + 2) * np.sum(pos / (n - ks))

    # 7) df & p‐value (detrend via AR(ar_order))
    npar = 2 * ar_order
    df = p * q * L - npar
    pval = chi2.sf(Q, df)

    return cons_flag_X, cons_flag_Y, Q, df, pval


# ─────────────────────────────────────────────────────────────────────────────
# 3) Worker for pre‐whitening (AR‐residuals)
def _compute_residual(i):
    ts = _dset[i]  # shape = (T, 1 + vars)
    T, nv1 = ts.shape
    nv = nv1 - 1
    # pre‐allocate
    out = np.empty((T - _ar_order, nv), float)
    for j in range(nv):
        out[:, j] = AutoReg(ts[:, j + 1], lags=_ar_order, old_names=False).fit().resid
    times = ts[_ar_order:, 0:1]
    return np.hstack((times, out))


# 4) Worker for CCF + linear‐test per‐gridbox
def _compute_stats_for_i(i):
    # extract the i-th residual time series (drop the time‐column)
    Xi = _res[i, :, 1:]  # shape = (n_times, n_vars)
    nbrs = _nbrs[i]
    n, p = Xi.shape

    ccf_pvals_i = []
    linear_pvals_i = []

    # reproducible jitter per gridbox
    rng = np.random.default_rng(1952 + i)

    for j in nbrs:
        # neighbor j
        Xj = _res[j, :, 1:]  # shape = (n_times, n_vars)

        # — cross‐portmanteau test
        fX, fY, Q, df, p_ccf = cross_lb_fft_test(
            Xi, Xj, m=int(round(n ** (1 / 3))), tol=_tol, ar_order=_ar_order
        )
        ccf_pvals_i.append([fX, fY, p_ccf])

        # — Pillai’s‐trace linear test
        tol_i = _tol * np.maximum(np.std(Xi, axis=0, keepdims=True), 1)
        tol_j = _tol * np.maximum(np.std(Xj, axis=0, keepdims=True), 1)
        Xi_j = Xi + rng.normal(0.0, tol_i, size=Xi.shape)
        Xj_j = Xj + rng.normal(0.0, tol_j, size=Xj.shape)

        exog = add_constant(Xj_j)
        endog = Xi_j
        J = np.eye(n) - np.ones((n, n)) / n
        B = np.linalg.pinv(exog) @ endog
        Bj = B[1:, :]
        H = Bj.T @ (exog[:, 1:].T @ J @ exog[:, 1:]) @ Bj
        E = (endog - exog @ B).T @ (endog - exog @ B)
        eigs = np.linalg.eigvals(np.linalg.inv(E) @ H)
        V = np.sum(eigs.real / (1 + eigs.real))
        num = (2 * n - 2 * _n_vars - 1) * V
        den = (_n_vars**2) * (1 - V)
        Fval = num / den
        p_lin = 1 - stats.f.cdf(Fval, _n_vars**2, 2 * n - 2 * _n_vars - 1)
        linear_pvals_i.append(p_lin)

    return i, ccf_pvals_i, linear_pvals_i


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_name", help="basename (no .npy)")
    p.add_argument("-p", "--ar_order", type=int, default=1)
    p.add_argument("-j", "--jobs", type=int, default=None)
    args = p.parse_args()

    # # workers
    n_jobs = args.jobs or cpu_count()
    print(f"Using {n_jobs} workers")

    # ── load & normalize ──────────────────────────────────────────────────────
    global _dset, _ar_order
    _ar_order = args.ar_order
    _dset = np.load(f"{args.data_name}_processed.npy")
    coords = np.load(f"{args.data_name}_coords_filtered.npy")
    # normalize each variable by its grand‐mean
    _dset[..., 1:] /= _dset[..., 1:].mean(axis=(0, 1))[None, None, :]

    # ── neighbor‐list via KDTree ───────────────────────────────────────────────
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    r_min = np.ceil(dists[:, 1].max())
    ball = tree.query_ball_point(coords, r_min)
    neighbors = [[j for j in nbrs if j != i] for i, nbrs in enumerate(ball)]
    print("Built neighbor_list (N=%d)" % len(neighbors))

    # saving list of neighbors for each point
    with open(f"{args.data_name}_neighbors.pkl", "wb") as f:
        pickle.dump(neighbors, f)

    # ── 3) Pre‐whitening in parallel ───────────────────────────────────────────
    N = _dset.shape[0]
    with Pool(n_jobs) as pool:
        resid_list = pool.map(_compute_residual, range(N))
    residuals = np.stack(resid_list, axis=0)
    del _dset
    print("Done with pre-processing data.")

    # ── 4) CCF & linear‐test in parallel ──────────────────────────────────────
    global _res, _nbrs, _tol, _n_vars
    _res = residuals
    _nbrs = neighbors
    _tol = 1e-8
    _n_vars = residuals.shape[2] - 1

    with Pool(n_jobs) as pool:
        out = pool.map(_compute_stats_for_i, range(N))

    # re‐assemble results
    CCFs_pvals = [None] * N
    linear_pvals = [None] * N
    for i, ccfv, linv in out:
        CCFs_pvals[i] = ccfv
        linear_pvals[i] = linv

    # save to disk
    with open(f"{args.data_name}_CCF_test.pkl", "wb") as f:
        pickle.dump(CCFs_pvals, f)
    with open(f"{args.data_name}_linear_test.pkl", "wb") as f:
        pickle.dump(linear_pvals, f)

    print("Done with cross-correlation analysis.")


if __name__ == "__main__":
    main()
