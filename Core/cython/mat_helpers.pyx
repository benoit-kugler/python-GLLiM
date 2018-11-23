cimport cython
cimport numpy as np
import numpy as np

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
cdef void dot_T_matrix(const double[:,:] A, const double[:,:] B, double[:,:] out) nogil:
    """write A.T * B + out in out"""
    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t K = A.shape[1]
    cdef Py_ssize_t M = B.shape[1]
    cdef Py_ssize_t i,j,k

    for i in range(K):
        for j in range(M):
            for k in range(N):
                out[i,j] += A[i,k] * B[k,j]


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
cdef void reset_zeros(double [:,:] A) nogil:
    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t M = A.shape[1]
    cdef Py_ssize_t i,j

    for i in range(N):
        for j in range(N):
            A[i,j] = 0
