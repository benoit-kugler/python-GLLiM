cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport isfinite, exp
from cython.parallel import prange

include "probas.pyx"


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_mult_mat(const double s, double[:,:] N) nogil:
    cdef Py_ssize_t n = N.shape[0]
    cdef Py_ssize_t i,j

    for i in range(n):
        for j in range(n):
            N[i,j] *= s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_mult_vect(const double s, double[:] V) nogil:
    cdef Py_ssize_t n = V.shape[0]
    cdef Py_ssize_t i

    for i in range(n):
        V[i] *= s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint isVfinite(const double[:] V) nogil:
    cdef Py_ssize_t n = V.shape[0]
    cdef Py_ssize_t i

    for i in range(n):
        if not isfinite(V[i]):
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_sum_vect( double[:] v_in, double s, double[:] out) nogil:
    """Write out + s * v_in ; in out"""
    cdef Py_ssize_t D = v_in.shape[0]
    cdef Py_ssize_t d
    for d in range(D):
        out[d] += v_in[d] * s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void scalar_sum_mat( double[:,:] v_in, double s, double[:,:] out) nogil:
    """Write out + s * v_in ; in out"""
    cdef Py_ssize_t D = v_in.shape[0]
    cdef Py_ssize_t d1, d2
    for d1 in range(D):
        for d2 in range(D):
            out[d1,d2] += v_in[d1,d2] * s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mu_helper_clean_add(const double[:,:] FXsi, const long[:] maski,
                    const double[:] wsi, const double[:] Yobsi,
                    double[:] esp_mu, double[:] maximal_mu) nogil:
    """Mean (Y - F) * w, with cleaning NaN values, added to maximal_mu"""
    cdef Py_ssize_t Ns = FXsi.shape[0]
    cdef Py_ssize_t D =  FXsi.shape[1]

    cdef double sum = 0
    cdef Py_ssize_t j, d

    for d in range(D):
        esp_mu[d] = 0

    for j in range(Ns):

        if (not maski[j]) and isfinite(wsi[j]) and isVfinite(FXsi[j]):
            sum += wsi[j]
            for d in range(D):
                esp_mu[d] += wsi[j] * (Yobsi[d] - FXsi[j,d])

    if not sum == 0:
        scalar_sum_vect(esp_mu, 1. / sum, maximal_mu)


@cython.boundscheck(False)
@cython.wraparound(False)
def mu_step_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_muNp = np.zeros(D)
    wsNp = np.ones((Ny,Ns)) / Ns
    maximal_muNp = np.zeros(D)

    cdef double[:] esp_mu = esp_muNp
    cdef double[:,:] ws = wsNp
    cdef double[:] maximal_mu = maximal_muNp


    cdef Py_ssize_t i, j, d
    cdef double sum = 0

    for i in range(Ny):
        mu_helper_clean_add(FXs[i],mask[i],ws[i],Yobs[i], esp_mu, maximal_mu)

    scalar_mult_vect(1. / Ny,maximal_mu)
    return maximal_muNp


@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_step_full_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask, const double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigmaNp = np.zeros((D, D))
    wsiNp = np.ones((Ny,Ns)) / Ns
    maximal_sigmaNp = np.zeros((D, D))
    tmpNp = np.zeros(D)

    cdef Py_ssize_t i,j, d1, d2, d
    cdef double sum = 0
    cdef double[:] tmp = tmpNp
    cdef double[:,:] esp_sigma = esp_sigmaNp
    cdef double[:,:] wsi = wsiNp
    cdef double[:,:] maximal_sigma = maximal_sigmaNp

    for i in range(Ny):
        sum = 0
        for d1 in range(D):
            tmp[d1] = 0
            for d2 in range(D):
                esp_sigma[d1,d2] = 0

        for j in range(Ns):

            if (not mask[i,j]) and isfinite(wsi[i,j]) and isVfinite(FXs[i,j]):
                sum += wsi[i,j]

                for d in range(D):
                    tmp[d] = FXs[i,j,d] + maximal_mu[d] - Yobs[i,d]

                for d1 in range(D):
                    for d2 in range(D):
                        esp_sigma[d1,d2] += wsi[i,j] * tmp[d1] * tmp[d2]

        if not sum == 0:
            scalar_sum_mat(esp_sigma, 1. / sum, maximal_sigma)

    scalar_mult_mat(1. / Ny, maximal_sigma)

    return maximal_sigmaNp


@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_step_diag_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask, const double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigmaNp = np.zeros(D)
    wsiNp = np.ones((Ny,Ns)) / Ns
    maximal_sigmaNp = np.zeros(D)

    cdef double[:] esp_sigma = esp_sigmaNp
    cdef double[:,:] wsi = wsiNp
    cdef double[:] maximal_sigma = maximal_sigmaNp

    cdef Py_ssize_t i,j, d
    cdef double sum = 0
    cdef double tmp = 0

    for i in range(Ny):
        tmp = 0
        sum = 0
        for d in range(D):
            esp_sigma[d] = 0

        for j in range(Ns):

            if (not mask[i,j]) and isfinite(wsi[i,j]) and isVfinite(FXs[i,j]):
                sum += wsi[i,j]

                for d in range(D):
                    tmp = FXs[i,j,d] + maximal_mu[d] - Yobs[i,d]
                    tmp = tmp ** 2
                    esp_sigma[d] += wsi[i,j] * tmp

        if not sum == 0:
            scalar_sum_vect(esp_sigma, 1. / sum, maximal_sigma)

    scalar_mult_vect(1. / Ny, maximal_sigma)

    return maximal_sigma


# ------------------------------ WITH IS ------------------------------ #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _helper_mu(const double[:,:] X, const double[:] weights, const double[:,:] means,
                const double[:,:,:] gllim_chol_covs, const double[:] log_p_tilde,
                const double[:,:] FX, const long[:] mask_x, const double[:] y,
                double[:] esp_mu, double[:] maximal_mu, double[:] wsi_view):

    q = densite_melange_precomputed(X, weights, means, gllim_chol_covs)

    cdef Py_ssize_t Ns = q.shape[0]

    cdef double[:] q_view = q

    cdef Py_ssize_t n

    for n in range(Ns):
        wsi_view[n] = exp(log_p_tilde[n]) / q_view[n] # Calcul des poids

    mu_helper_clean_add(FX, mask_x, wsi_view, y, esp_mu, maximal_mu)


@cython.boundscheck(False)
@cython.wraparound(False)
def mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]
    cdef Py_ssize_t K =  gllim_covs.shape[0]
    cdef Py_ssize_t L =  gllim_covs.shape[1]

    esp_muNp = np.zeros(D)
    wsNp = np.zeros((Ny,Ns))
    maximal_muNp = np.zeros(D)
    log_p_tilde = np.zeros(Ns)

    cdef double[:] esp_mu = esp_muNp
    cdef double[:,:] ws = wsNp
    cdef double[:] maximal_mu = maximal_muNp

    cdef Py_ssize_t i,  d, k, d1, d2
    cdef double sum = 0

    y_moins_mu = np.zeros(D)
    cdef double[:] y_moins_mu_view = y_moins_mu

    gllim_chol_covs = np.zeros((K, L, L))
    for k in range(K):
        gllim_chol_covs[k] = np.linalg.cholesky(gllim_covs[k])


    for i in range(Ny):
        for d in range(D):
            y_moins_mu_view[d] = Yobs[i,d] - current_mean[d]

        loggauspdf_diag(FXs[i],y_moins_mu,current_cov, log_p_tilde)
        _helper_mu(Xs[i], weightss[i], meanss[i], gllim_chol_covs, log_p_tilde,
                   FXs[i], mask[i], Yobs[i], esp_mu, maximal_mu, ws[i])
        # mu_helper_clean_add(FXs[i],mask[i],ws[i],Yobs[i], esp_mu, maximal_mu)

    scalar_mult_vect(1. / Ny,maximal_mu)
    return maximal_muNp, wsNp


def test(X, mu, cov):
    out = np.zeros(X.shape[0])
    cdef double[:] out_view = out
    loggauspdf_diag(X, mu, cov, out_view)
    return out
