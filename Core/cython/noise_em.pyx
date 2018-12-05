import multiprocessing
cimport cython
cimport openmp
cimport numpy as np
import numpy as np
from libc.math cimport isfinite, exp
from libc.stdio cimport printf
from cython.parallel import prange
from cython cimport view

include "probas.pyx"

cdef Py_ssize_t NUM_THREADS = multiprocessing.cpu_count()
if NUM_THREADS > 0:
    NUM_THREADS -= 1


# --------------------------- Helpers --------------------------- #

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
cdef void mu_helper_clean(const double[:,:] FXsi, const long[:] maski,
                    const double[:] wsi, const double[:] Yobsi,
                    double[:] tmp_mu) nogil:
    """Mean (Y - F) * w, with cleaning NaN values, wrote in tmp_mu"""
    cdef Py_ssize_t Ns = FXsi.shape[0]
    cdef Py_ssize_t D =  FXsi.shape[1]

    cdef double sum = 0
    cdef Py_ssize_t j, d

    for d in range(D):
        tmp_mu[d] = 0

    for j in range(Ns):

        if (not maski[j]) and isfinite(wsi[j]) and isVfinite(FXsi[j]):
            sum += wsi[j]
            for d in range(D):
                tmp_mu[d] += wsi[j] * (Yobsi[d] - FXsi[j,d])

    if not sum == 0:
        scalar_mult_vect( 1. / sum, tmp_mu)


# ------------------------------ Runners (C funcs) ------------------------------ #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_sigma_step_full(const double[:,:] Yobs, const double[:,:,:] FXs,
                          const long[:,:] mask, const double[:] maximal_mu,
                          const double[:,:] ws, double[:] tmp,
                          double[:,:] tmp_sigma, double[:,:] sigma_out) nogil:
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    cdef Py_ssize_t i,j, d1, d2, d
    cdef double sum = 0

    for i in range(Ny):
        sum = 0
        for d1 in range(D):
            tmp[d1] = 0
            for d2 in range(D):
                tmp_sigma[d1,d2] = 0

        for j in range(Ns):

            if (not mask[i,j]) and isfinite(ws[i,j]) and isVfinite(FXs[i,j]):
                sum += ws[i,j]

                for d in range(D):
                    tmp[d] = FXs[i,j,d] + maximal_mu[d] - Yobs[i,d]

                for d1 in range(D):
                    for d2 in range(D):
                        tmp_sigma[d1,d2] += ws[i,j] * tmp[d1] * tmp[d2]

        if not sum == 0:
            scalar_sum_mat(tmp_sigma, 1. / sum, sigma_out)

    scalar_mult_mat(1. / Ny, sigma_out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_sigma_step_diag(const double[:,:] Yobs, const double[:,:,:] FXs,
                               const long[:,:] mask, const double[:] maximal_mu,
                               const double[:,:] ws, double[:] tmp_sigma, double[:] sigma_out) nogil:
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]
    cdef Py_ssize_t i,j, d
    cdef double sum = 0
    cdef double tmp = 0

    for i in range(Ny):
        tmp = 0
        sum = 0
        for d in range(D):
            tmp_sigma[d] = 0

        for j in range(Ns):

            if (not mask[i,j]) and isfinite(ws[i,j]) and isVfinite(FXs[i,j]):
                sum += ws[i,j]

                for d in range(D):
                    tmp = FXs[i,j,d] + maximal_mu[d] - Yobs[i,d]
                    tmp = tmp ** 2
                    tmp_sigma[d] += ws[i,j] * tmp

        if not sum == 0:
            scalar_sum_vect(tmp_sigma, 1. / sum, sigma_out)

    scalar_mult_vect(1. / Ny, sigma_out)


#   WITH IS

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _helper_mu(const double[:,:] X, const double[:] weights, const double[:,:] means,
                const double[:,:,:] gllim_chol_covs, const double[:] log_p_tilde,
                const double[:,:] FX, const long[:] mask_x, const double[:] y,
                double[:] out_mu, double[:] out_weigths,
                double[:,:] tmp_KNs, double[:,:] tmp_KL) nogil:
    """Compute weights and esp_mu. Write in out_mu and out_weigths."""
    # cdef double[:] q_view = densite_melange_precomputed(X, weights, means, gllim_chol_covs)
    densite_melange_precomputed(X, weights, means, gllim_chol_covs, tmp_KNs, tmp_KL, out_weigths)

    cdef Py_ssize_t Ns = out_weigths.shape[0]

    cdef Py_ssize_t n
    cdef double max_log = log_p_tilde[0]
    cdef double sum = 0

    for n in range(Ns):   # re scaling to avoid numerical issue - finding max log
        if max_log < log_p_tilde[n]:
            max_log = log_p_tilde[n]


    for n in range(Ns):
        # if out_weigths[n] == 0:
        #     print(out_weigths[n])

        out_weigths[n] = exp(log_p_tilde[n] - max_log) / out_weigths[n] # Calcul des poids

    mu_helper_clean(FX, mask_x, out_weigths, y, out_mu)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_C_mu_step_diag_IS(const double[:,:] Yobs, const double[:,:,:] Xs,
                                    const double[:,:,:] meanss, const double[:,:] weightss,
                                    const double[:,:,:] FXs, const long[:,:] mask,
                                 const double[:] current_mean,
                                    const double[:] current_cov, double[:,:] y_moins_mu_view,
                                    double[:,:] log_p_tilde, const double[:,:,:] gllim_chol_covs,
                                    double[:,:] tmp_mu, double[:] maximal_mu, double[:,:] ws,
                                    double[:,:,:] tmp_KNs, double[:,:,:] tmp_KL) nogil:
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    cdef Py_ssize_t i, d, thread_number, ns

    for i in prange(Ny, nogil=True, num_threads=NUM_THREADS):
        thread_number = openmp.omp_get_thread_num()

        for d in range(D):
            y_moins_mu_view[thread_number,d] = Yobs[i,d] - current_mean[d]

        loggauspdf_diag(FXs[i],y_moins_mu_view[thread_number], current_cov, log_p_tilde[thread_number])

        _helper_mu(Xs[i], weightss[i], meanss[i], gllim_chol_covs, log_p_tilde[thread_number],
                   FXs[i], mask[i], Yobs[i], tmp_mu[thread_number], ws[i],
                   tmp_KNs[thread_number], tmp_KL[thread_number])

        for d in range(D):
            maximal_mu[d] += tmp_mu[thread_number,d]

    scalar_mult_vect(1. / Ny,maximal_mu)


def mu_step_diag_IS_i(np.ndarray yobs, const double[:,:] X,const double[:,:] means,
                      const double[:] weights, const double[:,:] FX, const long[:] mask_x,
                      np.ndarray current_mean, const double[:] current_cov,
                      const double[:,:,:] gllim_chol_covs):
    cdef Py_ssize_t Ns = FX.shape[0]
    cdef Py_ssize_t D =  FX.shape[1]
    cdef Py_ssize_t K = means.shape[0]
    cdef Py_ssize_t L = X.shape[1]

    cdef double[:] tmp_mu = np.zeros(D)
    cdef double[:] log_p_tilde = np.zeros(Ns)
    cdef double[:,:] tmp_KNs = np.zeros(( K,Ns))
    cdef double[:,:] tmp_KL = np.zeros(( K,L))
    cdef double[:] wsi = np.zeros(Ns)

    y_moins_mu = yobs - current_mean

    cdef double[:] y_moins_mu_view = y_moins_mu
    cdef double[:] y_obs_view = yobs


    with nogil:
        loggauspdf_diag(FX,y_moins_mu_view, current_cov, log_p_tilde)

        _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde,
                   FX, mask_x, y_obs_view, tmp_mu, wsi,
                   tmp_KNs, tmp_KL)

    return np.asarray(tmp_mu), np.asarray(wsi)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_C_mu_step_diag_IS_nopara(const double[:,:] Yobs, const double[:,:,:] Xs,
                                    const double[:,:,:] meanss, const double[:,:] weightss,
                                    const double[:,:,:] FXs, const long[:,:] mask,
                                        const double[:] current_mean,
                                    const double[:] current_cov, double[:] y_moins_mu_view,
                                    double[:] log_p_tilde, double[:,:,:] gllim_chol_covs,
                                    double[:] tmp_mu, double[:] maximal_mu, double[:,:] ws,
                                    double[:,:] tmp_KNs, double[:,:] tmp_KL) nogil:
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    cdef Py_ssize_t i, d, ns

    for i in range(Ny):

        for d in range(D):
            y_moins_mu_view[d] = Yobs[i,d] - current_mean[d]

        loggauspdf_diag(FXs[i],y_moins_mu_view, current_cov, log_p_tilde)

        _helper_mu(Xs[i], weightss[i], meanss[i], gllim_chol_covs, log_p_tilde,
                   FXs[i], mask[i], Yobs[i], tmp_mu, ws[i],
                   tmp_KNs, tmp_KL)

        for d in range(D):
            maximal_mu[d] += tmp_mu[d]

    scalar_mult_vect(1. / Ny,maximal_mu)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_C_mu_step_full_IS(const double[:,:] Yobs, const double[:,:,:] Xs,
                                    const double[:,:,:] meanss, const double[:,:] weightss,
                                    const double[:,:,:] FXs, const long[:,:] mask,
                                    const double[:,:,:] gllim_covs, const double[:] current_mean,
                                    const double[:,:] current_cov, double[:,:] y_moins_mu_view,
                                    double[:,:] log_p_tilde, double[:,:,:] gllim_chol_covs,
                                    double[:,:] tmp_mu, double[:] maximal_mu, double[:,:] ws,
                                    double[:,:,:] tmp_KNs, double[:,:,:] tmp_KL) nogil:
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t D =  FXs.shape[2]

    cdef Py_ssize_t i, d, thread_number

    for i in prange(Ny, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        for d in range(D):
            y_moins_mu_view[thread_number,d] = Yobs[i,d] - current_mean[d]

        chol_loggausspdf_precomputed(FXs[i],y_moins_mu_view[thread_number],current_cov, log_p_tilde[thread_number],
                                     tmp_mu[thread_number])
        _helper_mu(Xs[i], weightss[i], meanss[i], gllim_chol_covs, log_p_tilde[thread_number],
                   FXs[i], mask[i], Yobs[i], tmp_mu[thread_number], ws[i],
                   tmp_KNs[thread_number], tmp_KL[thread_number])

        for d in range(D):
            maximal_mu[d] += tmp_mu[thread_number,d]

    scalar_mult_vect(1. / Ny,maximal_mu)


def mu_step_full_IS_i(np.ndarray yobs, const double[:,:] X,const double[:,:] means,
                      const double[:] weights, const double[:,:] FX, const long[:] mask_x,
                      np.ndarray current_mean, const double[:,:] current_cov_chol,
                      const double[:,:,:] gllim_chol_covs):
    cdef Py_ssize_t Ns = FX.shape[0]
    cdef Py_ssize_t D =  FX.shape[1]
    cdef Py_ssize_t K = means.shape[0]
    cdef Py_ssize_t L = X.shape[1]

    cdef double[:] tmp_mu = np.zeros(D)
    cdef double[:] log_p_tilde = np.zeros(Ns)
    cdef double[:,:] tmp_KNs = np.zeros(( K,Ns))
    cdef double[:,:] tmp_KL = np.zeros(( K,L))
    cdef double[:] wsi = np.zeros(Ns)

    y_moins_mu = yobs - current_mean

    cdef double[:] y_moins_mu_view = y_moins_mu
    cdef double[:] y_obs_view = yobs

    with nogil:
        chol_loggausspdf_precomputed(FX,y_moins_mu_view,current_cov_chol, log_p_tilde,
                                     tmp_mu)

        _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde,
                   FX, mask_x, y_obs_view, tmp_mu, wsi,
                   tmp_KNs, tmp_KL)

    return np.asarray(tmp_mu), np.asarray(wsi)


# ----------------------------------- Entry points ----------------------------------- #

@cython.boundscheck(False)
@cython.wraparound(False)
def mu_step_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    tmp_muNp = np.zeros(D)
    wsNp = np.ones((Ny,Ns)) / Ns
    maximal_muNp = np.zeros(D)

    cdef double[:] tmp_mu = tmp_muNp
    cdef double[:,:] ws = wsNp
    cdef double[:] maximal_mu = maximal_muNp


    cdef Py_ssize_t i, j, d
    cdef double sum = 0

    for i in range(Ny):
        mu_helper_clean(FXs[i],mask[i],ws[i],Yobs[i], tmp_mu)

        for d in range(D):
            maximal_mu[d] += tmp_mu[d]

    scalar_mult_vect(1. / Ny,maximal_mu)
    return maximal_muNp



@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_step_full_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask, const double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    tmp_sigma = np.zeros((D, D))
    ws = np.ones((Ny,Ns)) / Ns
    maximal_sigma = np.zeros((D, D))
    tmp = np.zeros(D)

    _run_sigma_step_full(Yobs, FXs, mask, maximal_mu, ws, tmp, tmp_sigma, maximal_sigma)

    return maximal_sigma


@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_step_diag_NoIS(const double[:,:] Yobs, const double[:,:,:] FXs, const long[:,:] mask, const double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigmaNp = np.zeros(D)
    wsNp = np.ones((Ny,Ns)) / Ns
    maximal_sigmaNp = np.zeros(D)
    tmp_sigmaNp = np.zeros(D)

    _run_sigma_step_diag(Yobs, FXs, mask, maximal_mu, wsNp, tmp_sigmaNp, maximal_sigmaNp)

    return maximal_sigmaNp


# ---- WITH IS ---- #

@cython.boundscheck(False)
@cython.wraparound(False)
def mu_step_diag_IS(const double[:,:] Yobs, const double[:,:,:] Xs, const double[:,:,:] meanss,
                    const double[:,:] weightss, const double[:,:,:] FXs, const long[:,:] mask,
                    const double[:,:,:] gllim_covs, const double[:] current_mean,
                    const double[:] current_cov, parallel=True):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]
    cdef Py_ssize_t K =  gllim_covs.shape[0]
    cdef Py_ssize_t L =  gllim_covs.shape[1]


    ws = np.zeros((Ny,Ns))
    maximal_mu = np.zeros(D)
    gllim_chol_covs = np.zeros((K, L, L))
    for k in range(K):
        gllim_chol_covs[k] = np.linalg.cholesky(gllim_covs[k])


    if parallel:
        tmp_mu = np.zeros((NUM_THREADS,D))
        log_p_tilde = np.zeros((NUM_THREADS,Ns))
        tmp_KNs = np.zeros((NUM_THREADS, K,Ns))
        tmp_KL = np.zeros((NUM_THREADS, K,L))
        y_moins_mu = np.zeros((NUM_THREADS,D))
        _run_C_mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask,
                               current_mean, current_cov,
          y_moins_mu, log_p_tilde, gllim_chol_covs,
          tmp_mu, maximal_mu, ws, tmp_KNs, tmp_KL)
    else:
        tmp_mu = np.zeros(D)
        log_p_tilde = np.zeros(Ns)
        tmp_KNs = np.zeros(( K,Ns))
        tmp_KL = np.zeros(( K,L))
        y_moins_mu = np.zeros(D)
        _run_C_mu_step_diag_IS_nopara(Yobs, Xs, meanss, weightss, FXs, mask,
                                      current_mean, current_cov,
          y_moins_mu, log_p_tilde, gllim_chol_covs,
          tmp_mu, maximal_mu, ws, tmp_KNs, tmp_KL)

    return maximal_mu, ws


@cython.boundscheck(False)
@cython.wraparound(False)
def mu_step_full_IS(const double[:,:] Yobs, const double[:,:,:] Xs, const double[:,:,:] meanss,
                    const double[:,:] weightss, const double[:,:,:] FXs, const long[:,:] mask,
                    const double[:,:,:] gllim_covs, const double[:] current_mean,
                    const double[:,:] current_cov, parallel = None):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]
    cdef Py_ssize_t K =  gllim_covs.shape[0]
    cdef Py_ssize_t L =  gllim_covs.shape[1]

    tmp_mu = np.zeros((NUM_THREADS,D))
    ws = np.zeros((Ny,Ns))
    maximal_mu = np.zeros(D)
    log_p_tilde = np.zeros((NUM_THREADS,Ns))
    tmp_KNs = np.zeros((NUM_THREADS, K,Ns))
    tmp_KL = np.zeros((NUM_THREADS, K,L))
    y_moins_mu = np.zeros((NUM_THREADS,D))

    gllim_chol_covs = np.zeros((K, L, L))
    for k in range(K):
        gllim_chol_covs[k] = np.linalg.cholesky(gllim_covs[k])

    current_cov_chol = np.linalg.cholesky(current_cov)


    _run_C_mu_step_full_IS(Yobs, Xs, meanss, weightss, FXs, mask,
                              gllim_covs, current_mean, current_cov_chol,
                              y_moins_mu, log_p_tilde, gllim_chol_covs,
                              tmp_mu, maximal_mu, ws, tmp_KNs, tmp_KL)

    return maximal_mu, ws


def sigma_step_diag_IS(Yobs, FXs, ws, mask, maximal_mu):
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigmaNp = np.zeros(D)
    maximal_sigmaNp = np.zeros(D)
    tmp_sigmaNp = np.zeros(D)

    _run_sigma_step_diag(Yobs, FXs, mask, maximal_mu, ws, tmp_sigmaNp, maximal_sigmaNp)

    return maximal_sigmaNp


def sigma_step_full_IS(Yobs, FXs, ws, mask, maximal_mu):
    cdef Py_ssize_t D =  FXs.shape[2]

    tmp_sigma = np.zeros((D, D))
    maximal_sigma = np.zeros((D, D))
    tmp = np.zeros(D)

    _run_sigma_step_full(Yobs, FXs, mask, maximal_mu, ws, tmp, tmp_sigma, maximal_sigma)

    return maximal_sigma




# ------------------------------- DEBUG ------------------------------- #
def test(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y):
    Ns = FX.shape[0]
    K, L = means.shape
    tmp_mu = np.zeros(FX.shape[1])
    maximal_mu = np.zeros(FX.shape[1])
    wsi = np.zeros(FX.shape[0])
    tmp_KNs = np.zeros((K,Ns))
    tmp_KL = np.zeros((K,L))

    _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX,
               mask_x,y, tmp_mu, wsi, tmp_KNs, tmp_KL)
    return maximal_mu, wsi
