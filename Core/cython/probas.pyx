cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport isfinite, log, pi, exp, sqrt
from cython.parallel import prange
from cython.view cimport array


cdef double _LOG_2PI = log(2 * pi)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve_triangular_diff(const double[:,:] L, const double[:] X,
                                const double[:] mu, double[:] out) nogil:
    """Compute L-1 (X - mu), for X of shape (N,_)"""
    cdef Py_ssize_t D = X.shape[0]
    cdef Py_ssize_t d,i

    for d in range(D):
        out[d] = X[d] - mu[d]
        for i in range(d):
            out[d] -= L[d,i] * out[i]

        out[d] = out[d] / L[d,d]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void chol_loggausspdf_precomputed(const double[:,:] X, const double[:] mu,
                                  const double[:,:] cov_cholesky, double[:] out_view,
                                       double[:] tmp) nogil:
    """X : shape N,D"""
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t n,d

    for n in range(N):
        solve_triangular_diff(cov_cholesky, X[n], mu, tmp)
        out_view[n] = -0.5 * (D * _LOG_2PI )
        for d in range(D):
            out_view[n] += -0.5 * (tmp[d] ** 2) - log(cov_cholesky[d,d])



@cython.boundscheck(False)
@cython.wraparound(False)
def densite_melange_precomputed(const double[:,:] x_points, const double[:] weights,
                                const double[:,:] means, const double[:,:,:] chol_covs):
    """
    Compute the density of Gaussian mixture given at given points.
    :param x_points: shape (N,D)
    :param weights: shape (K,)
    :param means: shape (K,D)
    :param chol_covs: shape (K,D,D)
    :return: Array of densities, shape (N,)
    """
    cdef Py_ssize_t N = x_points.shape[0]
    cdef Py_ssize_t K = weights.shape[0]
    cdef Py_ssize_t D = means.shape[1]
    cdef Py_ssize_t k, n

    out = np.zeros(N)
    dens = np.zeros((K,N))
    tmp = np.zeros((K,D))

    cdef double[:] out_view = out
    cdef double[:,:] dens_view = dens
    cdef double[:,:] tmp_view = tmp

    for k in prange(K,nogil=True):
        chol_loggausspdf_precomputed(x_points, means[k], chol_covs[k], dens_view[k], tmp_view[k])
        for n in range(N):
            out_view[n] += exp(dens_view[k,n]) * weights[k]

    return out


# TODO pre compute sqrt(cov(d))
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void loggauspdf_diag(const double[:,:] X, const double[:] mu, const double[:] cov,
                      double[:] out) nogil:
    """Diagonal covariance matrix (cov is diagonal)
    X shape : N,D
    mu shape : D
    cov shape : D
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]

    cdef double log_det = 0
    cdef double q = 0
    cdef Py_ssize_t n,d

    for d in range(D):
        log_det += log(cov[d]) / 2

    for n in range(N):
        q = 0
        for d in range(D):
            q += ((X[n,d] - mu[d]) / sqrt(cov[d])) ** 2

        out[n] = -0.5 * (D * _LOG_2PI + q) - log_det
