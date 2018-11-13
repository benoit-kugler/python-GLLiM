cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void AdotBplusC(const double[:,:] A, const double[:] b, const double[:] c , double[:] out) nogil:
    cdef Py_ssize_t N = b.shape[0]

    cdef Py_ssize_t n,i
    cdef double s

    for n in range(N):
        s = 0
        for i in range(N):
           s += A[n,i] * b[i]
        s += c[n]
        out[n] = s



@cython.boundscheck(False)
@cython.wraparound(False)
def sampling_sameCov_chols(const double[:,:,:] means_list, const long[:,:] clusters_list,
                          const double[:,:,:] chols, const double[:,:] alea):
#    """Samples from N Gaussian Mixture Models
#
#    :param means_list: shape N,K,L
#    :param clusters_list: shape N,size
#    :param covs_list: shape K,L,L
#    :return: shape N,size,L
#    """
    cdef Py_ssize_t N = means_list.shape[0]
    cdef Py_ssize_t K = means_list.shape[1]
    cdef Py_ssize_t L = means_list.shape[2]
    cdef Py_ssize_t size = alea.shape[0]

    out = np.zeros((N, size, L))
    cdef double[:,:,:] out_view = out

    cdef Py_ssize_t n,s
    cdef long k

    for n in prange(N,nogil=True):
        for s in range(size):
            k = clusters_list[n][s]
#            out_view[n, s] = dot(chols[k],alea[s]) + means[k]
            AdotBplusC(chols[k],alea[s],means_list[n][k],out_view[n,s])
    return out