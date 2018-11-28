"""
Gllim model in cython. Implements parallelisation with respect to K.

__author__ = B. Kugler

The equation numbers refer to _High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables_A. Deleforge 2015

"""

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
import multiprocessing
cimport openmp
from cython.parallel import prange


include "probas.pyx"
include "mat_helpers.pyx"

# Constraint SFull : Sigma full; SDiag : Sigma diag; GFull : GammaT full; GDiag : GammaT diag
cdef double DEFAULT_REG_COVAR = 1e-08

cdef Py_ssize_t NUM_THREADS = multiprocessing.cpu_count()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _helper_rW_Z(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                 const double[:,:] Ak_T, const double[:] bk, const double[:] ck_W,
                 const double[:,:] ATSinv, double[:,:] ginv,
                 double[:,:] munk_W_out, double[:,:] Sk_W_out, double[:,:] Sk_X_out,
                 double[:,:] tmp_mat, double[:] tmp_vectD, double[:] tmp_vectLw) nogil:
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t Lt = Ak_T.shape[1]
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t n, d, l, l1, l2
    cdef Py_ssize_t L = Lt + Lw

    dot_matrix(ATSinv, Ak_W, ginv) # ginv + At * Sinv * Ak_W
    inverse_symetric(ginv, tmp_mat, Sk_W_out)

    for l1 in range(Lt):
        for l2 in range(L):
            Sk_X_out[l1,l2] = 0
    for l1 in range(Lt, L):
        for l2 in range(Lt):
            Sk_X_out[l1,l2] = 0
        for l2 in range(Lt, L):
            Sk_X_out[l1,l2] = Sk_W_out[l1 - Lt,l2 - Lt]


    for n in range(N):
        for d in range(D):
            tmp_vectD[d] = 0
        for l in range(Lw):
            tmp_vectLw[l] = 0

        dot(Ak_T,T[n],tmp_vectD)
        for d in range(D):
            tmp_vectD[d] = Y[n,d] - tmp_vectD[d] - bk[d]

        dot(ATSinv,tmp_vectD, tmp_vectLw)
        dot(ginv, ck_W, tmp_vectLw)
        dot(Sk_W_out, tmp_vectLw, munk_W_out[n])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GIso_SIso(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double Gammak_W, const double Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t l1, l2, d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W
            else:
                ginv_tmpLw[l1,l2] = 0

        for d in range(D):
            ATSinv_tmp[l1,d] = Ak_W[d,l1] / Sigmak

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GIso_SDiag(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double Gammak_W, const double[:] Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t l1, l2, l, d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return


    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W
            else:
                ginv_tmpLw[l1,l2] = 0


    for l in range(Lw):
        for d in range(D):
            ATSinv_tmp[l,d] = Ak_W[d,l] / Sigmak[d]

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GIso_SFull(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double Gammak_W, const double[:,:] Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t l1, l2

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W
            else:
                ginv_tmpLw[l1,l2] = 0


    inverse_symetric(Sigmak, tmp_DD, tmp_DD2)
    dot_T_matrix(Ak_W, tmp_DD2, ATSinv_tmp)

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GDiag_SIso(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double[:] Gammak_W, const double Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t l1, l2, d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W[l1]
            else:
                ginv_tmpLw[l1,l2] = 0

        for d in range(D):
            ATSinv_tmp[l1,d] = Ak_W[d,l1] / Sigmak


    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GDiag_SDiag(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double[:] Gammak_W, const double[:] Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t l1, l2, d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W[l1]
            else:
                ginv_tmpLw[l1,l2] = 0


        for d in range(D):
            ATSinv_tmp[l1,d] = Ak_W[d,l1] / Sigmak[d]


    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GDiag_SFull(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double[:] Gammak_W, const double[:,:] Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t l1, l2

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    for l1 in range(Lw):
        for l2 in range(Lw):
            if l1 == l2:
                ginv_tmpLw[l1,l2] = 1. / Gammak_W[l1]
            else:
                ginv_tmpLw[l1,l2] = 0

    inverse_symetric(Sigmak, tmp_DD, tmp_DD2)
    dot_T_matrix(Ak_W, tmp_DD2, ATSinv_tmp)

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GFull_SIso(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double[:,:] Gammak_W, const double Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t l1,d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    inverse_symetric(Gammak_W, tmp_LwLw, ginv_tmpLw)


    for l1 in range(Lw):
        for d in range(D):
            ATSinv_tmp[l1,d] = Ak_W[d,l1] / Sigmak


    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GFull_SDiag(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                            const double[:,:] Ak_T, const double[:,:] Gammak_W, const double[:] Sigmak,
                            const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                            double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                            double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                            double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t l1,d

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    inverse_symetric(Gammak_W, tmp_LwLw, ginv_tmpLw)

    for l1 in range(Lw):
        for d in range(D):
            ATSinv_tmp[l1,d] = Ak_W[d,l1] / Sigmak[d]


    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_rW_Z_GFull_SFull(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                                    const double[:,:] Ak_T, const double[:,:] Gammak_W, const double[:,:] Sigmak,
                                    const double[:] bk, const double[:] ck_W, double[:,:] munk_W_out,
                                    double[:,:] Sk_W_out, double[:,:] Sk_X_out, double[:] tmp_D, double[:] tmp_Lw,
                                    double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw, double[:,:] ATSinv_tmp,
                                    double[:,:] tmp_DD, double[:,:] tmp_DD2) nogil:
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    cdef Py_ssize_t Lw = Ak_W.shape[1]

    if Lw == 0:
        reset_zeros(Sk_X_out)
        return

    inverse_symetric(Gammak_W, tmp_LwLw, ginv_tmpLw)

    inverse_symetric(Sigmak, tmp_DD, tmp_DD2)
    dot_T_matrix(Ak_W, tmp_DD2, ATSinv_tmp)

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk, ck_W,
                 ATSinv_tmp, ginv_tmpLw, munk_W_out, Sk_W_out, Sk_X_out,
                 tmp_LwLw, tmp_D, tmp_Lw)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Xnk(const double[:,:] T, const double[:,:] munk, double[:,:] Xnk) nogil:
    cdef Py_ssize_t Lw = munk.shape[1]
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n, lt, lw

    for n in range(N):
        for lt in range(Lt):
            Xnk[n,lt] = T[n,lt]
        for lw in range(Lw):
            Xnk[n, Lt + lw] = munk[n,lw]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_ck_T(const double[:,:] T, const double[:]rnk, const double rk,
                        double[:] ck_T) nogil:
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n, l
    cdef double s

    for n in range(N):
        s = rnk[n] / rk
        for l in range(Lt):
            ck_T[l] += s * T[n,l]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_GammaT_GFull(const double[:,:] T, const double[:] rnk, const double rk, const double[:] ck_T,
                                double[:,:] Gammak_T, double[:] tmp) nogil:
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n, l1, l2
    cdef double s

    for n in range(N):
        s = sqrt(rnk[n] / rk)
        for l in range(Lt):
            tmp[l] = s * (T[n,l] - ck_T[l])

        for l1 in range(Lt):
            for l2 in range(Lt):
                Gammak_T[l1,l2] += tmp[l1] * tmp[l2]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _compute_GammaT_GIso(const double[:,:] T, const double[:] rnk, const double rk,
                               const double[:] ck_T) nogil:
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n
    cdef double s, tmp
    cdef double out = 0

    if Lt == 0:
        return 0

    for n in range(N):
        tmp = 0
        s = sqrt(rnk[n] / rk)
        for l in range(Lt):
            tmp += (s * (T[n,l] - ck_T[l])) ** 2
        tmp /= Lt
        out += tmp
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_GammaT_GDiag(const double[:,:] T, const double[:] rnk, const double rk,
                               const double[:] ck_T, double[:] Gammak_T) nogil:
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n
    cdef double s

    for n in range(N):
        s = sqrt(rnk[n] / rk)
        for l in range(Lt):
            Gammak_T[l] += (s * (T[n,l] - ck_T[l])) ** 2



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _add_numerical_stability_full(double[:,:] M) nogil:
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t n

    for n in range(N):
        M[n,n] += DEFAULT_REG_COVAR


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _add_numerical_stability_diag(double[:] M) nogil:
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t n

    for n in range(N):
        M[n] += DEFAULT_REG_COVAR


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _add_numerical_stability_iso(const double M) nogil:
    return M + DEFAULT_REG_COVAR


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Ak(const double[:,:] Y, const double[:,:] Xnk, const double[:] rnk,
                      const double rk, double[:,:] Sk_X, double[:,:] Ak,
                      double[:] xk_bar, double[:] yk_bar, double[:,:] X_stark,
                      double[:,:] Y_stark, double[:,:] YXt_stark, double[:,:] inv_tmp) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t n,l,d, l2
    cdef double s = 0

    for n in range(N):
        s = rnk[n] / rk
        for l in range(L):
            xk_bar[l] += s * Xnk[n,l]

        for d in range(D):
            yk_bar[d] += s * Y[n,d]

    for n in range(N):
        s = sqrt(rnk[n] / rk)
        for l in range(L):
            X_stark[l,n] = s * (Xnk[n,l] -  xk_bar[l]) # (33)

        for d in range(D):
            Y_stark[d,n] = s  * (Y[n,d] - yk_bar[d])  # (34)

    MMt(X_stark, Sk_X) # Sk_X + Sk_X * Sk_Xt
    inverse_symetric_inplace(Sk_X, inv_tmp)  # (31)
    dot_matrix_T(Y_stark, X_stark, YXt_stark)
    dot_matrix(YXt_stark, inv_tmp, Ak)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_bk(const double[:,:] Y, const double[:,:] Xnk,
                const double[:] rnk, const double rk, const double[:,:] Ak,
                double[:] bk) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t n,l,d
    cdef double s, u

    for n in range(N):
        s = rnk[n] / rk
        for d in range(D):
            u = 0
            for l in range(L):
                u += Ak[d,l] * Xnk[n,l]
            bk[d] += s * (Y[n,d] - u)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Sigmak_SFull(const double[:,:] Y, const double[:,:] Xnk, const double[:]rnk,
                          const double rk, const double[:,:] Ak, const double[:] bk,
                          const double[:,:] Sk_W, double[:] tmp, double[:,:] Sigmak) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t Lw = Sk_W.shape[0]
    cdef Py_ssize_t Lt = L - Lw

    cdef Py_ssize_t n,l,d, d1, d2, i, j
    cdef double s, u

    for n in range(N):
        s = sqrt(rnk[n] / rk)
        for d in range(D):
            u = 0
            for l in range(L):
                u += Ak[d,l] * Xnk[n,l]
            tmp[d] = s * (Y[n,d] - u - bk[d])

        for d1 in range(D):
            for d2 in range(D):
                Sigmak[d1,d2] += tmp[d1] * tmp[d2]

    for d1 in range(D):
        for d2 in range(D):
            for i in range(Lw):
                for j in range(Lw):  # Ak_W
                    Sigmak[d1,d2] += Ak[d1, i + Lt] * Sk_W[i,j] * Ak[d2,j + Lt]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Sigmak_SDiag(const double[:,:] Y, const double[:,:] Xnk, const double[:]rnk,
                          const double rk, const double[:,:] Ak, const double[:] bk,
                          const double[:,:] Sk_W, double[:] Sigmak) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t Lw = Sk_W.shape[0]
    cdef Py_ssize_t Lt = L - Lw

    cdef Py_ssize_t n,l,d, i ,j
    cdef double s, u

    for n in range(N):
        s = rnk[n] / rk
        for d in range(D):
            u = 0
            for l in range(L):
                u += Ak[d,l] * Xnk[n,l]
            Sigmak[d] += s * ((Y[n,d] - u - bk[d]) ** 2)

    for d in range(D):
        for i in range(Lw):
            for j in range(Lw):  # Ak_W
                Sigmak[d] += Ak[d, i + Lt] * Sk_W[i,j] * Ak[d,j + Lt]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _compute_Sigmak_SIso(const double[:,:] Y, const double[:,:] Xnk, const double[:]rnk,
                                 const double rk, const double[:,:] Ak, const double[:] bk,
                                 const double[:,:] Sk_W) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]
    cdef Py_ssize_t Lw = Sk_W.shape[0]
    cdef Py_ssize_t Lt = L - Lw

    cdef Py_ssize_t n,l,d, i, j
    cdef double s, u
    cdef double Sigmak = 0

    for n in range(N):
        s = sqrt(rnk[n] / rk)
        for d in range(D):
            u = 0
            for l in range(L):
                u += Ak[d,l] * Xnk[n,l]
            Sigmak += (s * (Y[n,d] - u - bk[d])) ** 2

    for d in range(D):
        for i in range(Lw):
            for j in range(Lw):  # Ak_W
                Sigmak += Ak[d, i + Lt] * Sk_W[i,j] * Ak[d,j + Lt]

    Sigmak /= D
    return Sigmak


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void resets(double[:,:] munk, double[:,:] inv_tmp, double[:,:] YXt_stark, double[:,:] Sk_W,
                 double[:,:] ATSinv_tmp, double[:,:] tmp_LwLw, double[:,:] ginv_tmpLw,
                 double[:] xk_bar, double[:] yk_bar, double[:] tmp_D, double[:] tmp_Lw) nogil:
    cdef Py_ssize_t N = munk.shape[0]
    cdef Py_ssize_t Lw = munk.shape[1]
    cdef Py_ssize_t L = xk_bar.shape[0]
    cdef Py_ssize_t D = yk_bar.shape[0]

    cdef Py_ssize_t n, l, l2, d

    for n in range(N):
        for l in range(Lw):
            munk[n,l] = 0

    for l in range(L):
        xk_bar[l] = 0
        for l2 in range(L):
            inv_tmp[l,l2] = 0

    for d in range(D):
        yk_bar[d] = 0
        tmp_D[d] = 0
        for l in range(L):
            YXt_stark[d,l] = 0

    for l in range(Lw):
        tmp_Lw[l] = 0
        for d in range(D):
            ATSinv_tmp[l,d] = 0
        for l2 in range(Lw):
            Sk_W[l,l2] = 0
            ginv_tmpLw[l,l2] = 0
            if tmp_LwLw is not None:
                tmp_LwLw[l,l2] = 0



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GIso_SIso(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                             const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:] GammakList_W,
                             const double[:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                             double[:] out_pikList, double[:,:] out_ckList_T, double[:] out_GammakList_T,
                             double[:,:,:] out_AkList, double[:,:] out_bkList, double[:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]


        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GIso_SIso(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])
        out_GammakList_T[k] = _compute_GammaT_GIso(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k])
        out_GammakList_T[k] = _add_numerical_stability_iso(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        out_SigmakList[k] = _compute_Sigmak_SIso(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number])
        out_SigmakList[k] = _add_numerical_stability_iso(out_SigmakList[k])


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GIso_SDiag(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                             const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:] GammakList_W,
                             const double[:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                             double[:] out_pikList, double[:,:] out_ckList_T, double[:] out_GammakList_T,
                             double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:] out_SigmakList,
                                  double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]


        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GIso_SDiag(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        out_GammakList_T[k] = _compute_GammaT_GIso(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k])
        out_GammakList_T[k] = _add_numerical_stability_iso(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SDiag(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], out_SigmakList[k])
        _add_numerical_stability_diag(out_SigmakList[k])


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GIso_SFull(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:] GammakList_W,
                                 const double[:,:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:,:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GIso_SFull(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        out_GammakList_T[k] = _compute_GammaT_GIso(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k])
        out_GammakList_T[k] = _add_numerical_stability_iso(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SFull(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], tmp_D[thread_number], out_SigmakList[k])
        _add_numerical_stability_full(out_SigmakList[k])


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GDiag_SIso(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:] GammakList_W,
                                 const double[:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GDiag_SIso(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GDiag(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k])
        _add_numerical_stability_diag(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        out_SigmakList[k] = _compute_Sigmak_SIso(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number])
        out_SigmakList[k] = _add_numerical_stability_iso(out_SigmakList[k])



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GDiag_SDiag(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:] GammakList_W,
                                 const double[:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GDiag_SDiag(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GDiag(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k])
        _add_numerical_stability_diag(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SDiag(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], out_SigmakList[k])
        _add_numerical_stability_diag(out_SigmakList[k])


# ----------------------- Gamma Diag and Sigma Full ----------------------- #
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GDiag_SFull(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:] GammakList_W,
                                 const double[:,:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:,:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GDiag_SFull(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GDiag(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k])
        _add_numerical_stability_diag(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SFull(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], tmp_D[thread_number], out_SigmakList[k])
        _add_numerical_stability_full(out_SigmakList[k])



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GFull_SIso(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:,:] GammakList_W,
                                 const double[:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GFull_SIso(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GFull(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k], tmp_Lt[thread_number])
        _add_numerical_stability_full(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        out_SigmakList[k] = _compute_Sigmak_SIso(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number])
        out_SigmakList[k] = _add_numerical_stability_iso(out_SigmakList[k])



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GFull_SDiag(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:,:] GammakList_W,
                                 const double[:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:] out_SigmakList,
                                double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                               double[:] rk):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]

        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GFull_SDiag(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GFull(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k], tmp_Lt[thread_number])
        _add_numerical_stability_full(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SDiag(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], out_SigmakList[k])
        _add_numerical_stability_diag(out_SigmakList[k])


# ----------------------- Gamma Full and Sigma Full ----------------------- #
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_next_theta_GFull_SFull(const double[:,:] T, const double[:,:] Y, const double[:,:] rnk_List,
                                 const double[:,:,:] AkList_W, const double[:,:,:] AkList_T, const double[:,:,:] GammakList_W,
                                 const double[:,:,:] SigmakList, const double[:,:] bkList, const double[:,:] ckList_W,
                                 double[:] out_pikList, double[:,:] out_ckList_T, double[:,:,:] out_GammakList_T,
                                 double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:,:] out_SigmakList,
                                 double[:,:,:] munk, double[:,:,:] Sk_W, double[:,:,:] Sk_X,
                                 double[:,:,:] Xnk, double[:,:] tmp_Lt, double[:,:] tmp_D,
                                 double[:,:] xk_bar, double[:,:] yk_bar, double[:,:,:] X_stark,
                                 double[:,:,:] Y_stark, double[:,:,:] YXt_stark, double[:,:,:] ATSinv_tmp,
                                 double[:,:,:] inv_tmp, double[:,:] tmp_Lw, double[:,:,:] tmp_LwLw,
                                 double[:,:,:] tmp_DD, double[:,:,:] tmp_DD2, double[:,:,:] ginv_tmpLw,
                                   double[:] rk ):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]

    cdef Py_ssize_t n, k, thread_number

    for k in prange(K, nogil=True, num_threads=NUM_THREADS, schedule='static'):
        thread_number = openmp.omp_get_thread_num()

        rk[thread_number] = 0
        for n in range(N):
            rk[thread_number] += rnk_List[n,k]


        resets(munk[thread_number], inv_tmp[thread_number], YXt_stark[thread_number], Sk_W[thread_number],
               ATSinv_tmp[thread_number], tmp_LwLw[thread_number], ginv_tmpLw[thread_number],
               xk_bar[thread_number], yk_bar[thread_number], tmp_D[thread_number], tmp_Lw[thread_number])  # tmp memory set to zero

        out_pikList[k] = rk[thread_number] / N

        _compute_rW_Z_GFull_SFull(Y,T, AkList_W[k], AkList_T[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk[thread_number], Sk_W[thread_number], Sk_X[thread_number], tmp_D[thread_number],
                                  tmp_Lw[thread_number], tmp_LwLw[thread_number],ginv_tmpLw[thread_number],
                                  ATSinv_tmp[thread_number], tmp_DD[thread_number], tmp_DD2[thread_number])

        _compute_ck_T(T, rnk_List[:,k], rk[thread_number], out_ckList_T[k])

        _compute_GammaT_GFull(T, rnk_List[:,k], rk[thread_number],  out_ckList_T[k], out_GammakList_T[k], tmp_Lt[thread_number])
        _add_numerical_stability_full(out_GammakList_T[k])

        _compute_Xnk(T, munk[thread_number], Xnk[thread_number])

        _compute_Ak(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], Sk_X[thread_number], out_AkList[k],
                    xk_bar[thread_number],  yk_bar[thread_number],  X_stark[thread_number],
                    Y_stark[thread_number],  YXt_stark[thread_number],  inv_tmp[thread_number])

        _compute_bk(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k])

        _compute_Sigmak_SFull(Y, Xnk[thread_number], rnk_List[:,k], rk[thread_number], out_AkList[k], out_bkList[k],
                              Sk_W[thread_number], tmp_D[thread_number], out_SigmakList[k])
        _add_numerical_stability_full(out_SigmakList[k])
