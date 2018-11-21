import multiprocessing
cimport cython
cimport openmp
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdio cimport printf
from cython.parallel import prange

# Constraint SFull : Sigma full; SDiag : Sigma diag; GFull : GammaT full; GDiag : GammaT diag

include "probas.pyx"

cdef double DEFAULT_REG_COVAR = 1e-08



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dot(const double [:,:] M, const double[:] v, double[:] out) nogil:
    """out = out + Mv"""
    cdef Py_ssize_t D = M.shape[0]
    cdef Py_ssize_t L = M.shape[1]
    cdef Py_ssize_t d,l

    for d in range(D):
        for l in range(L):
            out[d] += M[d,l] * v[l]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dot_matrix(const double[:,:] A, const double[:,:] B, double[:,:] out) nogil:
    """write AB + out in out"""
    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t K = A.shape[1]
    cdef Py_ssize_t M = B.shape[1]
    cdef Py_ssize_t i,j,k

    for i in range(N):
        for j in range(M):
            for k in range(K):
                out[i,j] += A[i,k] * B[k,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dot_matrix_T(const double[:,:] A, const double[:,:] B, double[:,:] out) nogil:
    """write A * B.T + out in out"""
    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t K = A.shape[1]
    cdef Py_ssize_t M = B.shape[0]
    cdef Py_ssize_t i,j,k

    for i in range(N):
        for j in range(M):
            for k in range(K):
                out[i,j] += A[i,k] * B[j,k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void MMt(const double[:,:] A, double[:,:] out) nogil:
    """write M * transpose(M) + out in out"""
    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t M = A.shape[1]
    cdef Py_ssize_t i,j,k

    for i in range(N):
        for j in range(N):
            for k in range(M):
                out[i,j] += A[i,k] * A[j,k]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _helper_rW_Z(const double[:,:] Y, const double[:,:] T, const double[:,:] Ak_W,
                 const double[:,:] Ak_T, const double[:] bk, const double[:] ck_W,
                 const double[:,:] ATSinv, double[:,:] ginv,
                 double[:,:] munk_W_out, double[:,:] Sk_W_out,
                 double[:,:] tmp_mat, double[:] tmp_vectD, double[:] tmp_vectLw) nogil:
    cdef Py_ssize_t Lw = Ak_W.shape[1]
    cdef Py_ssize_t Lt = Ak_T.shape[1]
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t n, d, l


    dot_matrix(ATSinv, Ak_W, ginv) # ginv + At * Sinv * Ak_W
    inverse_symetric(ginv, tmp_mat, Sk_W_out)

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




def _compute_rW_Z_GFull_SFull(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.linalg.inv(Gammak_W)
    ATSinv = np.matmul(Ak_W.T, np.linalg.inv(Sigmak))
    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GDiag_SFull(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.diag(1 / Gammak_W)
    ATSinv = np.matmul(Ak_W.T, np.linalg.inv(Sigmak))

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)

def _compute_rW_Z_GIso_SFull(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = (1 / Gammak_W) * np.eye(Lw)
    ATSinv = np.matmul(Ak_W.T, np.linalg.inv(Sigmak))

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GFull_SDiag(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.linalg.inv(Gammak_W)
    ATSinv = np.matmul(Ak_W.T, np.diag(Sigmak))
    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GDiag_SDiag(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.diag(1 / Gammak_W)
    ATSinv = np.matmul(Ak_W.T, np.diag(Sigmak))

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GIso_SDiag(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = (1 / Gammak_W) * np.eye(Lw)
    ATSinv = np.matmul(Ak_W.T, np.diag(Sigmak))

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GFull_SIso(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.linalg.inv(Gammak_W)
    ATSinv = (1 / Sigmak) *  Ak_W.T
    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GDiag_SIso(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = np.diag(1 / Gammak_W)
    ATSinv = (1 / Sigmak) *  Ak_W.T

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


def _compute_rW_Z_GIso_SIso(Y, T, Ak_W, Ak_T, Gammak_W, Sigmak, bk, ck_W,
                        munk_W_out, Sk_W_out):
    """
    Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W
    write munk_W_out shape (N,Lw) Sk_W_out shape (Lw,Lw)
    """
    D, Lw = Ak_W.shape

    if Lw == 0:
        return

    tmp_mat = np.zeros((Lw,Lw))
    tmp_vectD = np.zeros(D)
    tmp_vectLw = np.zeros(Lw)

    ginv = (1 / Gammak_W) * np.eye(Lw)
    ATSinv = (1 / Sigmak) *  Ak_W.T

    _helper_rW_Z(Y, T, Ak_W, Ak_T, bk,ck_W,
                 ATSinv, ginv, munk_W_out, Sk_W_out,
                 tmp_mat, tmp_vectD, tmp_vectLw)


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
            Xnk[n, lw + Lt] = munk[n,lw]

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
cdef void _compute_GammaT_GIso(const double[:,:] T, const double[:] rnk, const double rk,
                               const double[:] ck_T, double Gammak_T) nogil:
    cdef Py_ssize_t N = T.shape[0]
    cdef Py_ssize_t Lt = T.shape[1]

    cdef Py_ssize_t n
    cdef double s, tmp

    for n in range(N):
        tmp = 0
        s = sqrt(rnk[n] / rk)
        for l in range(Lt):
            tmp += (s * (T[n,l] - ck_T[l])) ** 2
        tmp /= Lt
        Gammak_T += tmp

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
cdef void _add_numerical_stability_iso(double M) nogil:
    M += DEFAULT_REG_COVAR


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Ak(const double[:,:] Y, const double[:,:] Xnk, const double[:] rnk,
                      const double rk, double[:,:] Sk_X, double[:,:] Ak,
                      double[:] xk_bar, double[:] yk_bar, double[:,:] X_stark,
                      double[:,:] Y_stark, double[:,:] YXt_stark, double[:,:] inv) nogil:
    cdef Py_ssize_t N = Y.shape[0]
    cdef Py_ssize_t L = Xnk.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t n,l,d
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
    inverse_symetric_inplace(Sk_X, inv)  # (31)

    dot_matrix_T(Y_stark, X_stark, YXt_stark)
    dot_matrix(YXt_stark, inv, Ak)


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
        s = sqrt(rnk[n] / rk)
        for d in range(D):
            u = 0
            for l in range(L):
                u += Ak[d,l] * Xnk[n,l]
            Sigmak[d] += (s * (Y[n,d] - u - bk[d])) ** 2

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
cdef void compute_next_theta_GFull_SFull(const double[:,:] T, const double[:,:] Y, const long Lw,
                                         const double[:,:] rnk_List, const double[:,:,:] AkList_W,
                                         const double[:,:,:] GammakList_W, const double[:,:,:] SigmakList,
                                         const double[:,:] bkList, const double[:,:] ckList_W,
                                         double[:] out_pikList, double[:,:] out_ckList_T, double[:,:,:] out_GammakList_T,
                                         double[:,:,:] out_AkList, double[:,:] out_bkList, double[:,:,:] out_SigmakList,
                                         double[:,:] munk, double[:,:] Sk_W, double[:,:] Xnk, double[:] tmp_Lt,
                                         double[:] xk_bar, double[:] yk_bar, double[:,:] X_stark,
                                         double[:,:] Y_stark, double[:,:] YXt_stark, double[:,:] inv,
                                         double[:] tmp_D):
    cdef Py_ssize_t N = rnk_List.shape[0]
    cdef Py_ssize_t K = rnk_List.shape[1]
    cdef Py_ssize_t Lt = T.shape[1]
    cdef Py_ssize_t D = Y.shape[1]

    cdef Py_ssize_t k
    cdef double rk

    for k in range(K):
        rk = 0
        for n in range(N):
            rk += rnk_List[n,k]

        out_pikList[k] = rk / N

        _compute_rW_Z_GFull_SFull(Y,T, AkList_W[k], GammakList_W[k],
                                  SigmakList[k], bkList[k], ckList_W[k],
                                  munk, Sk_W)

        _compute_Xnk(T, munk, Xnk)

        _compute_ck_T(T, rnk_List[k], rk, out_ckList_T[k])

        _compute_GammaT_GFull(T, rnk_List[k], rk,  out_ckList_T[k], out_GammakList_T[k], tmp_Lt)
        _add_numerical_stability_full(out_GammakList_T[k])

        _compute_Ak(Y, Xnk, rnk_List[k], rk, Sk_W, out_AkList[k],
                    xk_bar,  yk_bar,  X_stark, Y_stark,  YXt_stark,  inv)

        _compute_bk(Y, Xnk, rnk_List[k], rk, out_AkList[k], out_bkList[k])

        _compute_Sigmak_SFull(Y, Xnk, rnk_List[k], rk, out_AkList[k], out_bkList[k],
                              Sk_W, tmp_D, out_SigmakList[k])
        _add_numerical_stability_full(out_SigmakList[k])





def test_complet(T, Y, rnk_List, AkList_W, GammakList_W,  SigmakList, bkList,  ckList_W):
    N,D = Y.shape
    K ,_, Lw = AkList_W.shape
    _, Lt = T.shape
    L = Lt + Lw

    out_pikList = np.zeros(K)
    out_ckList_T = np.zeros((K, Lt))
    out_GammakList_T = np.zeros((K, Lt, Lt))
    out_AkList = np.zeros((K, D, L))
    out_bkList = np.zeros((K, D))
    out_SigmakList = np.zeros((K,D,D))

    xk_bar = np.zeros(L)
    yk_bar = np.zeros(D)
    X_stark = np.zeros((L,N))
    Y_stark = np.zeros((D,N))
    YXt_stark = np.zeros((D,L))
    inv = np.zeros((L,L))

    munk = np.zeros((N,Lw))  # tmp
    Sk_W = np.zeros((Lw,Lw))  # tmp
    Xnk = np.zeros((N,L)) # tmp
    tmp_Lt = np.zeros(Lt) # tmp
    tmp_D = np.zeros(D) # tmp

    compute_next_theta_GFull_SFull(T, Y, Lw, rnk_List, AkList_W, GammakList_W,  SigmakList, bkList,  ckList_W,
                                   out_pikList,  out_ckList_T,  out_GammakList_T, out_AkList,  out_bkList,
                                   out_SigmakList, munk,  Sk_W,  Xnk,  tmp_Lt, xk_bar,  yk_bar,  X_stark,
                                   Y_stark,  YXt_stark,  inv, tmp_D)

    return out_pikList, out_ckList_T, out_GammakList_T, out_AkList, out_bkList, out_SigmakList
