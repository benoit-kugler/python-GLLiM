import logging
import time

import numpy as np
import numba as nb
from scipy import linalg

from Core import cython

_LOG_2PI = np.log(2 * np.pi)

# log of pdf for gaussian distributuion with diagonal covariance matrix
def loggausspdf(X, mu, cov):
    if len(X.shape)==1:
        D=1
    else:
        D = X.shape[1]

    logDetCov = D*np.log(cov)
    dxM = X - mu
    L = np.sqrt(cov)
    xRinv = 1/L * dxM
    mahalaDx = np.sum(xRinv**2, axis=1)
    y = - 0.5 * (logDetCov + D*_LOG_2PI + mahalaDx)
    return y

def gausspdf(X, mu, cov):
    return np.exp(loggausspdf(X, mu, cov))


def chol_loggausspdf(X, mu, cov):
    if mu.ndim == 2:
        return _chol_loggausspdf2(X, mu, cov)
    else:
        return _chol_loggausspdf(X, mu, cov)


def chol_loggausspdf_precomputed(X, mu, cov):
    if mu.ndim == 2:
        return _chol_loggausspdf_precomputed2(X, mu, cov)
    else:
        return _chol_loggausspdf_precomputed(X, mu, cov)


def chol_loggausspdf_iso(X, mu, cov):
    if mu.ndim == 2:
        return _chol_loggausspdf_iso2(X, mu, cov)
    else:
        return _chol_loggausspdf_iso(X, mu, cov)


def chol_loggauspdf_diag(X, mu, cov):
    if mu.ndim == 2:
        return _chol_loggauspdf_diag2(X, mu, cov)
    else:
        return _chol_loggauspdf_diag(X, mu, cov)


@nb.njit(nogil=True, fastmath=True, cache=True)
def _chol_loggausspdf(X, mu, cov):
    """log of pdf for gaussian distributuion with full covariance matrix
    (cholesky factorization for stability)
    X shape : D,N
    mu shape : D
    cov shape : D,D
    """
    D, N = X.shape
    mu = np.copy(mu).reshape((-1, 1))
    X = X - mu  # DxN
    U = np.linalg.cholesky(cov)  # DxD
    Q = np.linalg.solve(U, X)
    q = np.sum(Q ** 2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(nogil=True, fastmath=True, cache=True)
def _chol_loggausspdf2(X, mu, cov):
    """log of pdf for gaussian distributuion with full covariance matrix
    (cholesky factorization for stability)
    X shape : D,N
    mu shape : D,N
    cov shape : D,D
    """
    D, N = X.shape
    X = X - mu  # DxN
    U = np.linalg.cholesky(cov)  # DxD
    Q = np.linalg.solve(U, X)
    q = np.sum(Q ** 2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(nogil=True, fastmath=True, cache=True)
def _chol_loggausspdf_precomputed(X, mu, cov_cholesky):
    """log of pdf for gaussian distributuion with full covariance matrix
    (cholesky factorization for stability)
    if cholesky is given, use cholesky.T instead of computing it.
    X shape : D,N
    mu shape : D
    cov shape : D,D
    """
    D, N = X.shape
    mu2 = np.copy(mu).reshape((-1, 1))
    X = X - mu2  # DxN
    U = cov_cholesky  # DxD
    Q = np.linalg.solve(U, X)
    q = np.sum(Q ** 2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit((nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]), nogil=True, fastmath=True, cache=True)
def _chol_loggausspdf_precomputed2(X, mu, cov_cholesky):
    """log of pdf for gaussian distributuion with full covariance matrix (cholesky factorization for stability)
    X shape : D,N
    mu shape : D,N
    cov shape : D,D
    """
    D,N = X.shape
    X = X - mu #DxN
    U = cov_cholesky  # DxD
    Q = np.linalg.solve(U,X)
    q = np.sum(Q**2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(nogil=True, cache=True)
def _chol_loggausspdf_iso(X, mu, cov):
    """Spherical covariance matrix (cov is scalar)
    X shape : D,N
    mu shape : D or D,N
    cov shape : ()
    """
    D, N = X.shape
    mu2 = np.copy(mu).reshape((-1, 1))
    X = X - mu2  # DxN
    Q = X / np.sqrt(cov)
    q = np.sum(Q ** 2, axis=0)
    log_det = D * np.log(cov) / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(nogil=True, cache=True)
def _chol_loggausspdf_iso2(X, mu, cov):
    """Spherical covariance matrix (cov is scalar)
    X shape : D,N
    mu shape : D,N
    cov shape : ()
    """
    D,N = X.shape
    X = X - mu #DxN
    Q = X / np.sqrt(cov)
    q = np.sum(Q**2, axis=0)
    log_det = D * np.log(cov) / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(cache=True)
def _chol_loggauspdf_diag(X, mu, cov):
    """Diagonal covariance matrix (cov is diagonal)
    X shape : D,N
    mu shape : D
    cov shape : D
    """
    D, N = X.shape
    mu2 = np.copy(mu).reshape((-1, 1))
    X = X - mu2  # DxN
    Q = X / np.sqrt(cov.reshape((-1, 1)))
    q = np.sum(Q ** 2, axis=0)
    log_det = np.log(cov).sum() / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.njit(cache=True)
def _chol_loggauspdf_diag2(X, mu, cov):
    """Diagonal covariance matrix (cov is diagonal)
    X shape : D,N
    mu shape : D,N
    cov shape : D
    """
    D, N = X.shape
    X = X - mu  # DxN
    Q = X / np.sqrt(cov.reshape((-1,1)))
    q = np.sum(Q ** 2, axis=0)
    log_det = np.log(cov).sum() / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


@nb.jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def densite_melange(x_points, weights, means, covs):
    """
    Compute the density of Gaussian mixture given at given points.
    :param x_points: shape (N,L)
    :param weights: shape (K,)
    :param means: shape (K,L)
    :param covs: shape (K,L,L)
    :return: Array of densities, shape (N,)
    """
    N = x_points.shape[0]
    K = weights.shape[0]
    r = np.empty((K, N))
    for i in range(len(means)):
        s = _chol_loggausspdf(x_points.T, means[i], covs[i])
        r[i] = np.exp(s) * weights[i]
    return np.sum(r, axis=0)


@nb.jit(nopython=True, nogil=True, fastmath=True)
def densite_melange_precomputed(x_points, weights, means, chol_covs):
    """
    Compute the density of Gaussian mixture given at given points.
    :param x_points: shape (N,L)
    :param weights: shape (K,)
    :param means: shape (K,L)
    :param chol_covs: shape (K,L,L)
    :return: Array of densities, shape (N,)
    """
    N = x_points.shape[0]
    K = weights.shape[0]
    r = np.zeros((K, N))
    for i in range(len(means)):
        s = _chol_loggausspdf_precomputed(x_points.T, means[i], chol_covs[i])
        r[i] = np.exp(s) * weights[i]
    return np.sum(r, axis=0)


@nb.njit(nogil=True, cache=True)
def cholesky_list(Cs):
    N, D, D = Cs.shape
    out = np.zeros((N, D, D))
    for i in range(N):
        out[i] = np.linalg.cholesky(Cs[i])
    return out


def covariance_melange(weigths,means,covs):
    """Returns a matrix of same shape as covs[0]"""
    sc = (weigths[:,None] * means).sum(axis=0)
    p1 = weigths[:,None,None] * (covs + np.matmul(means[:,:,None],means[:,None,:]))
    return p1.sum(axis=0) - (sc[None,:] * sc[:,None])


def dominant_components(weights,means,covs,threshold=None,sort_by="height",dets=None):
    """Returns a sorted list of parameters of mixture. Order is made on height or weight.
    If threshold is given, gets rid of components with weight <= threshold.
    If dets is given, use instead of re-computing det(covs)"""
    if dets is None:
        chols = np.linalg.cholesky(covs)
        dets = np.prod(np.array([np.diag(c) for c in chols]),axis=1)
    heights = weights / dets
    if threshold:
        mask = weights > threshold
        heights , weights, means, covs = heights[mask] , weights[mask] , means[mask] , covs[mask]
    i_sort = {"height":0,"weight":1}[sort_by]
    return sorted(zip(heights,weights,means,covs), key=lambda d: d[i_sort], reverse=True)


@nb.njit(cache=True, fastmath=True)
def _GMM_sampling_sameCov(means_list: np.ndarray, clusters_list: np.ndarray,
                          covs_list: np.ndarray, alea: np.ndarray):
    """Samples from N Gaussian Mixture Models

    :param means_list: shape N,K,L
    :param clusters_list: shape N,size
    :param covs_list: shape K,L,L
    :return: shape N,size,L
    """
    N, K, L = means_list.shape
    size, _ = alea.shape
    out = np.empty((N, size, L))
    chols = cholesky_list(covs_list)

    for n in range(N):
        clusters = clusters_list[n]
        means = means_list[n]
        for s in range(size):
            k = clusters[s]
            out[n, s] = chols[k].dot(alea[s]) + means[k]
    return out


@nb.njit(cache=True, fastmath=True)
def _GMM_sampling_Covs(means_list: np.ndarray, clusters_list: np.ndarray,
                       covs_list: np.ndarray, alea: np.ndarray):
    """Samples from N Gaussian Mixture Models

    :param means_list: shape N,K,L
    :param clusters_list: shape N,size
    :param covs_list: shape N,K,L,L
    :return: shape N,size,L
    """
    N, K, L = means_list.shape
    size, _ = alea.shape
    out = np.empty((N, size, L))

    for n in range(N):
        clusters = clusters_list[n]
        means = means_list[n]
        chols = cholesky_list(covs_list[n])
        for s in range(size):
            k = clusters[s]
            out[n, s] = chols[k].dot(alea[s]) + means[k]
    return out


def GMM_sampling(means_list: np.ndarray, weights_list: np.ndarray,
                 covs_list: np.ndarray, size: int):
    """Samples from N Gaussian Mixture Models

    :param means_list: shape N,K,L
    :param weights_list: shape N,K
    :param covs_list: shape N,K,L,L or shape K,L,L (same covs)
    :param size:
    :return:
    """
    _, _, L = means_list.shape
    alea = np.random.multivariate_normal(np.zeros(L), np.eye(L), size)

    clusters_list = cython.multinomial_sampling_cython(weights_list, size)

    if covs_list.ndim == 3:
        return _GMM_sampling_sameCov(means_list, clusters_list, covs_list, alea)
    else:
        return _GMM_sampling_Covs(means_list, clusters_list, covs_list, alea)


if __name__ == '__main__':
    # pik = np.arange(2) + 1
    # means = np.arange(2*3).reshape((2,3))
    # covs = np.arange(2*3*3).reshape((2,3,3))  + 1
    # cov = 3 * np.eye(3)
    D = 10
    T = np.tril(np.ones((D, D))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD
    X = np.random.random_sample((D, 100000))


    # print(covs)
    # print(covariance_melange(pik,means,covs))

    def f1():
        Q = np.linalg.solve(U.T, X)


    def f2():
        Q2 = linalg.solve_triangular(U.T, X, lower=True)

    #
    # assert np.allclose(Q,Q2)
    K = 40
    N = 200
    size = 100000
    covs = np.array([cov] * K)
    wks = np.random.random_sample((N, K))
    wks /= wks.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((N, K, D))

    GMM_sampling(meanss, wks, covs, size)