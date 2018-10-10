import logging
import time

import numpy as np
from scipy import linalg

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


def chol_loggausspdf(X, mu, cov, cholesky=None):
    """log of pdf for gaussian distributuion with full covariance matrix
    (cholesky factorization for stability)
    if cholesky is given, use cholesky.T instead of computing it.
    X shape : D,N
    mu shape : D
    cov shape : D,D
    """
    D,N = X.shape
    mu = np.atleast_2d(mu)
    X = X - mu #DxN
    U = np.linalg.cholesky(cov).T if cholesky is None else cholesky.T  # DxD
    Q = linalg.solve_triangular(U.T,X,lower=True)
    q = np.sum(Q**2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det


def chol_loggausspdf_iso(X,mu,cov):
    """Spherical covariance matrix (cov is scalar)
    X shape : D,N
    mu shape : D
    cov shape : ()"""
    D,N = X.shape
    mu = np.atleast_2d(mu)
    X = X - mu #DxN
    Q = X / np.sqrt(cov)
    q = np.sum(Q**2, axis=0)
    log_det = D * np.log(cov) / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


def chol_loggauspdf_diag(X, mu, cov):
    """Diagonal covariance matrix (cov is diagonal)
    X shape : D,N
    mu shape : D
    cov shape : D"""
    D, N = X.shape
    mu = np.atleast_2d(mu)
    X = X - mu  # DxN
    Q = X / np.sqrt(cov[:, None])
    q = np.sum(Q ** 2, axis=0)
    log_det = np.log(cov).sum() / 2
    return -0.5 * (D * _LOG_2PI + q) - log_det


def densite_melange(x_points,weights,means,covs):
    """
    Compute the density of Gaussian mixture given at given points.
    :param x_points: shape (N,L)
    :param weights: shape (K,)
    :param means: shape (K,L)
    :param covs: shape (K,L,L)
    :return: Array of densities, shape (N,)
    """
    r = [chol_loggausspdf(x_points.T,m[:,None],c)  for m,c in zip(means,covs) ]
    return np.sum(weights * np.exp(np.array(r).T) , axis = 1)

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
    return sorted(zip(heights,weights,means,covs),key=lambda d: d[i_sort],reverse=True)


def GMM_sampling(means_list: np.ndarray, weights_list: np.ndarray,
                 covs_list: np.ndarray, size: int):
    """Samples from N Gaussian Mixture Models

    :param means_list: shape N,K,L
    :param weights_list: shape N,K
    :param covs_list: shape N,K,L,L or shape K,L,L for N identical covs
    :param size: Number of sample per point
    :return: shape N,size,L
    """
    ti = time.time()
    N, K, L = means_list.shape
    out = np.empty((N, size, L))
    alea = np.random.multivariate_normal(np.zeros(L), np.eye(L), size)
    precompute_chols = (covs_list.ndim == 3)
    if precompute_chols:
        chols = np.linalg.cholesky(covs_list)

    for weights, means, n in zip(weights_list, means_list, range(N)):
        if not precompute_chols:
            chols = np.linalg.cholesky(covs_list[n])
        clusters = np.random.multinomial(1, weights, size=size).argmax(axis=1)
        means = np.array([means[k] for k in clusters])
        stds = np.array([chols[k] for k in clusters])
        out[n] = np.matmul(stds, alea[:, :, None])[:, :, 0] + means
    logging.debug(f"Sampling from mixture ({N} series of {size}) done in {time.time()-ti:.3f} s")
    return out


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
