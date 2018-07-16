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

# log of pdf for gaussian distributuion with full covariance matrix (cholesky factorization for stability)

def chol_loggausspdf(X, mu, cov):
    D,N = X.shape
    mu = np.atleast_2d(mu)
    X = X - mu #DxN
    U = np.linalg.cholesky(cov).T #DxD
    Q = linalg.solve_triangular(U.T,X,lower=True)
    q = np.sum(Q**2, axis=0)
    log_det = np.sum(np.log(np.diag(U)))
    return -0.5 * (D * _LOG_2PI + q) - log_det

def chol_loggausspdf_iso(X,mu,cov):
    """Spherical covariance matrix (cov is scalar)"""
    D,N = X.shape
    mu = np.atleast_2d(mu)
    X = X - mu #DxN
    Q = X / np.sqrt(cov)
    q = np.sum(Q**2, axis=0)
    log_det = D * np.log(cov) / 2
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



# pik = np.arange(2) + 1
# means = np.arange(2*3).reshape((2,3))
# covs = np.arange(2*3*3).reshape((2,3,3))  + 1
# cov = 3 * np.eye(3)
D = 10
T = np.tril(np.ones((D, D))) * 0.456
cov = np.dot(T, T.T)
U = np.linalg.cholesky(cov).T #DxD
X = np.random.random_sample((D,100000))
# print(covs)
# print(covariance_melange(pik,means,covs))

def f1():
    Q = np.linalg.solve(U.T, X)

def f2():
    Q2 = linalg.solve_triangular(U.T,X,lower=True)


#
# assert np.allclose(Q,Q2)

