cimport
cython
cimport
numpy as np
import numpy as np

np.import_array()

DTYPE = np.float
ctypedef
np.float_t
DTYPE_t


def Hapke_vect(np.


    ndarray
SZA, np.ndarray
VZA, np.ndarray
DPHI, np.ndarray
W, np.ndarray
R1,
np.ndarray
BB, np.ndarray
CC, np.ndarray
HH, np.ndarray
B0):
cdef
float
pi = np.pi
cdef
np.ndarray
theta0r = SZA * pi / 180
cdef
np.ndarray
thetar = VZA * pi / 180
cdef
np.ndarray
phir = DPHI * pi / 180
cdef
np.ndarray
R = R1 * pi / 180
return theta0r
