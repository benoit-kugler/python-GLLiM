cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport isfinite
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sMplusN_mat(const double s, const double[:,:] M, double[:,:] N) nogil:
    # inplace N = N + sM
    cdef Py_ssize_t n = M.shape[0]
    cdef Py_ssize_t i,j

    for i in range(n):
        for j in range(n):
            N[i,j] += s * M[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sMplusN_vect(const double s, const double[:] M, double[:] N) nogil:
    # inplace N = N + sM
    cdef Py_ssize_t n = M.shape[0]
    cdef Py_ssize_t i

    for i in range(n):
        N[i] += s * M[i]


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
cdef bint isMfinite(const double[:,:] M) nogil:
    cdef Py_ssize_t n = M.shape[0]
    cdef Py_ssize_t i,j

    for i in range(n):
        for j in range(n):
            if not isfinite(M[i,j]):
                return False
    return True


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
cdef void clean_mean_matrix(const double[:,:,:] Gx, const double[:] w, const long[:] mask_x, double[:,:] out) nogil:
    cdef Py_ssize_t N = Gx.shape[0]
    cdef int L = Gx.shape[1]

    cdef double s, poids
    s = 0
    cdef Py_ssize_t i

    for i in range(N):
        poids = w[i]
        if (not mask_x[i]) and isfinite(poids) and isMfinite(Gx[i]):
            s += poids
            sMplusN_mat(poids,Gx[i],out)
    if s == 0:
        scalar_mult_mat(0, out)
    else:
        scalar_mult_mat(1/s, out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void clean_mean_vector(const double[:,:] Gx, const double[:] w, const long[:] mask_x, double[:] out) nogil:
    cdef Py_ssize_t N = Gx.shape[0]
    cdef int L = Gx.shape[1]

    cdef double s, poids
    s = 0
    cdef Py_ssize_t i

    for i in range(N):
        poids = w[i]
        if (not mask_x[i]) and isfinite(poids) and isVfinite(Gx[i]):
            s += poids
            sMplusN_vect(poids,Gx[i],out)

    if s == 0:
        scalar_mult_vect(0, out)
    else:
        scalar_mult_vect(1/s, out)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vect_vectT(const double[:] u, double[:,:] out) nogil:
    cdef Py_ssize_t n = u.shape[0]
    cdef Py_ssize_t i,j

    for i in range(n):
        for j in range(n):
            out[i,j] = u[i] * u[j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vect_add(const double[:] u1, const double[:] u2, const double[:] u3, double[:] out) nogil:
    # u1 + u2 - u3
    cdef Py_ssize_t n = u1.shape[0]
    cdef Py_ssize_t i

    for i in range(n):
        out[i] = u1[i] + u2[i] - u3[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_clean(double[:,:] max_sigma, double[:,:] current_esp_sigma) nogil:
    cdef Py_ssize_t D = max_sigma.shape[0]
    cdef Py_ssize_t i,j

    for i in range(D):
        for j in range(D):
            max_sigma[i,j] += current_esp_sigma[i,j]
            current_esp_sigma[i,j] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_clean_vector(double[:] max_sigma, double[:] current_esp_sigma) nogil:
    cdef Py_ssize_t D = max_sigma.shape[0]
    cdef Py_ssize_t i

    for i in range(D):
        max_sigma[i] += current_esp_sigma[i]
        current_esp_sigma[i] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_sigma_step_full_NoIS(const double[:,:,:] FXs, const double[:] maximal_mu,
        const double[:,:] Yobs, double[:] u, double[:,:,:] G3, const double[:] wsi,
        const long[:,:] mask, double[:,:] esp_sigma, double[:,:] maximal_sigma) nogil:
    cdef Py_ssize_t i,j
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]

    for i in range(Ny):
        for j in range(Ns):
            vect_add(FXs[i,j],maximal_mu,Yobs[i], u)
            vect_vectT(u,G3[j])
        clean_mean_matrix(G3, wsi, mask[i], esp_sigma)
        add_clean(maximal_sigma, esp_sigma)
    scalar_mult_mat(1. / Ny, maximal_sigma)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void square(const double[:] u, double[:] out) nogil:
    cdef Py_ssize_t N = u.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        out[i] = u[i] ** 2



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _run_sigma_step_diag_NoIS(const double[:,:,:] FXs, const double[:] maximal_mu,
        const double[:,:] Yobs, double[:] u, double[:,:] G3, const double[:] wsi,
        const long[:,:] mask, double[:] esp_sigma, double[:] maximal_sigma) nogil:
    cdef Py_ssize_t i,j
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]

    for i in range(Ny):
        for j in range(Ns):
            vect_add(FXs[i,j],maximal_mu,Yobs[i], u)
            square(u,G3[j])

        clean_mean_vector(G3, wsi, mask[i], esp_sigma)
        add_clean_vector(maximal_sigma, esp_sigma)
    scalar_mult_vect(1. / Ny, maximal_sigma)


def sigma_step_full_NoIS(double[:,:] Yobs, double[:,:,:] FXs, long[:,:] mask, double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigma = np.zeros((D, D))
    G3 = np.zeros((Ns, D, D))
    u = np.zeros(D)
    wsi = np.ones(Ns) / Ns
    maximal_sigma = np.zeros((D, D))

    _run_sigma_step_full_NoIS(FXs, maximal_mu, Yobs, u, G3, wsi, mask, esp_sigma, maximal_sigma)
    return maximal_sigma


def sigma_step_diag_NoIS(double[:,:] Yobs, double[:,:,:] FXs, long[:,:] mask, double[:] maximal_mu):
    cdef Py_ssize_t Ny = FXs.shape[0]
    cdef Py_ssize_t Ns = FXs.shape[1]
    cdef Py_ssize_t D =  FXs.shape[2]

    esp_sigma = np.zeros(D)
    G3 = np.zeros((Ns, D))
    u = np.zeros(D)
    wsi = np.ones(Ns) / Ns
    maximal_sigma = np.zeros(D)

    _run_sigma_step_diag_NoIS(FXs, maximal_mu, Yobs, u, G3, wsi, mask, esp_sigma, maximal_sigma)
    return maximal_sigma