cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange




#cdef multinomial_sampling(const double[:] weights, int size) nogil:
#    cdef Py_ssize_t K = weights.shape[1]
#
#    cumweights = np.cumsum(weights)
#    alea = np.random.random_sample(size)
#
#    out = np.empty(size,dtype=np.int)
#    cdef long[:] out_view = out
#    cdef double[:] S = cumweights
#    cdef double[:] A = alea
#
#    cdef Py_ssize_t i
#    cdef long k
#    cdef double current_alea
#
#    for i in range(size):
#        current_alea = A[i]
#        k = 0
#        while k < K and current_alea > S[k]:
#            k = k + 1
#        out_view[i] = k + 1
#    return out
#
#
#
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def GMM_sampling(double[:,:,:] means_list, double[:,:] weights_list,
#                 double[:,:,:] covs_list, int size):
#    cdef Py_ssize_t N = means_list.shape[0]
#    cdef Py_ssize_t K = means_list.shape[1]
#    cdef Py_ssize_t L = means_list.shape[2]
#
#    out = np.zeros((N,size, L),dtype = np.double)
#    cdef double[:,:,:] out_view = out
#    cdef double[:] v
#    cdef const long[:] clusters
#    cdef const double[:,:] means, alea_normal
#
#    alea_normal = np.random.multivariate_normal(np.zeros(L), np.eye(L), size)
#    #precompute chols
#    chols = np.linalg.cholesky(covs_list)
#
#
#    cdef Py_ssize_t n,i,k
#    for n in prange(N,nogil=True):
#        clusters = multinomial_sampling(weights_list[n],size)
#        means = means_list[n]
#        for i in range(size):
#            k = clusters[i]
#            v = chols[k].dot(alea_normal[i]) + means[k]
#            out_view[n,i,:] = v
#    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def multinomial_sampling(const double[:,:] weights, int size):
    cdef Py_ssize_t N = weights.shape[0]
    cdef Py_ssize_t K = weights.shape[1]

    cumweights = np.cumsum(weights,axis=1)
    alea = np.random.random_sample((N,size))

    out = np.empty((N,size),dtype=np.int)
    cdef long[:,:] out_view = out
    cdef double[:,:] S = cumweights
    cdef double[:,:] A = alea

    cdef Py_ssize_t n,i
    cdef long k
    cdef double current_alea

    for n in prange(N,nogil=True):
        for i in range(size):
            current_alea = A[n,i]
            k = 0
            while k < K and current_alea > S[n,k]:
                k = k + 1
            out_view[n,i] = k
    return out