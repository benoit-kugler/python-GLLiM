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
    """Compute L-1 (X - mu), for X of shape (N,_). L is lower.
    Erase out.
    """
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
    """X : shape N,D
    erase out_view and tmp
    """
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
cdef void chol_loggausspdf2_precomputed(const double[:,:] X, const double[:,:] mu,
                                  const double[:,:] cov_cholesky, double[:] out_view,
                                       double[:] tmp) nogil:
    """X : shape N,D
    mu shape : N,D
    erase out_view and tmp
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t D = X.shape[1]
    cdef Py_ssize_t n,d

    for n in range(N):
        solve_triangular_diff(cov_cholesky, X[n], mu[n], tmp)
        out_view[n] = -0.5 * (D * _LOG_2PI )
        for d in range(D):
            out_view[n] += -0.5 * (tmp[d] ** 2) - log(cov_cholesky[d,d])





@cython.boundscheck(False)
@cython.wraparound(False)
cdef void densite_melange_precomputed(const double[:,:] x_points, const double[:] weights,
                                const double[:,:] means, const double[:,:,:] chol_covs,
                                double[:,:] tmp_KN, double[:,:] tmp_KL, double[:] out) nogil:
    """
    Compute the density of Gaussian mixture given at given points.
    :param x_points: shape (N,L)
    :param weights: shape (K,)
    :param means: shape (K,L)
    :param chol_covs: shape (K,L,L)
    Write in out. erase tmp_KN, tmp_KL
    """
    cdef Py_ssize_t N = x_points.shape[0]
    cdef Py_ssize_t K = weights.shape[0]
    cdef Py_ssize_t L = means.shape[1]
    cdef Py_ssize_t k, n

    for k in range(K):
        chol_loggausspdf_precomputed(x_points, means[k], chol_covs[k], tmp_KN[k], tmp_KL[k])
        for n in range(N):
            out[n] += exp(tmp_KN[k,n]) * weights[k]





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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void loggauspdf_iso(const double[:,:] X, const double[:] mu, double cov,
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
        log_det += log(cov) / 2

    for n in range(N):
        q = 0
        for d in range(D):
            q += ((X[n,d] - mu[d]) / sqrt(cov)) ** 2

        out[n] = -0.5 * (D * _LOG_2PI + q) - log_det

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cholesky(const double[:,:] A, double[:,:] L) nogil:
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    write the lower variant triangular matrix, L, which should be zero at first."""
    cdef Py_ssize_t n = A.shape[0]

    cdef double tmp_sum = 0
    cdef Py_ssize_t i,k,j

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i+1):
            tmp_sum = 0
            for j in range(k):
                tmp_sum += L[i,j] * L[k,j]

            if (i == k): # Diagonal elements
                L[i,k] = sqrt(A[i,i] - tmp_sum)
            else:
                L[i,k] = (1. / L[k,k] * (A[i,k] - tmp_sum))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inverse_triangular(const double[:,:] L, double[:,:] out) nogil:
    """matrix inversion"""
    cdef Py_ssize_t n = L.shape[0]

    cdef Py_ssize_t i,j,k

    for i in range(n):
        out[i,i] = 1. / L[i,i]
        for j in range(i): # j < i
            out[i,j] = 0
            out[j,i] = 0
            for k in range(j,i):
                out[i, j] +=  L[i,k] * out[k,j]
            out[i,j] /= - L[i,i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void tAA_low_tri(const double[:,:] A, double[:,:] out) nogil:
    """Work for A lower triangular. Write transpose(A) * A """
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i,j,k

    for i in range(n):
        for j in range(i+1):
            out[i,j] = 0
            for k in range(i,n):
                out[i,j] += A[k,i] * A[k,j]
            out[j,i] = out[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inverse_symetric(const double[:,:] S, double[:,:] tmp, double[:,:] out) nogil:
    """cholesky decomp + triangular inversion. Write on S"""
    cholesky(S, out)
    inverse_triangular(out, tmp)
    tAA_low_tri(tmp, out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inverse_symetric_inplace(double[:,:] S, double[:,:] out) nogil:
    """cholesky decomp + triangular inversion. Write on S"""
    cholesky(S, out)
    inverse_triangular(out, S)
    tAA_low_tri(S, out)


def test_chol(A):
    L = np.zeros(A.shape)
    M = np.zeros(A.shape)
    inverse_symetric_inplace(A,L)

    return L
