# cython: profile=True

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport exp, tan, cos, sin, log1p, sqrt, acos
from cython.parallel import prange

np.import_array()

cdef int VARIANT = 2002

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef (double,double,double,double,double,double,double,double) T8Double
ctypedef (double,double,double) T3Double
ctypedef (double,double,double, double, double) T5Double

cdef double pi = np.pi


#cdef double e1(double R, double x):
#    return exp(-2 / (tan(R) * tan(x) * pi))
#
#
#cdef double e2(double R, double x):
#    return exp(-1 / (tan(R)**2 * tan(x)**2 * pi ))



cdef T5Double fast_e1_e2(double R,double x, double x0) nogil:
    cdef double tr, tr2, trtx, trtx0, e2x, e1x, e2x0 , e1x0,
    tr = tan(R)
    tr2 = tr**2
    trtx = tr * tan(x)
    trtx0 = tr * tan(x0)
    if trtx == 0:
        e1x = 0
        e2x = 0
    else:
        e1x = exp(-2 / (trtx * pi))
        e2x = exp(-1 / (trtx**2 * pi ))

    if trtx0 == 0:
        e1x0 = 0
        e2x0 = 0
    else:
        e1x0 = exp(-2 / (trtx0  * pi))
        e2x0 = exp(-1 / (trtx0 **2 * pi ))
    return e1x,e2x, e1x0, e2x0, tr


cdef double H_2002(double x,double y) nogil:
    return (1 + 2 * x) / (1 + 2 * x * y)



cdef double H_1993(double x,double y) nogil: # eq 3 from icarus Schimdt
    u = (1 - y) / (1 + y)
    return 1 / ( 1 - (1-y) * x * (  u  +  (1 - (0.5 + x) * u) * log1p( 1/x ) ))


cdef T8Double _geom_roughness(double theta0, double theta, double phi) nogil:
    """Only geometrical-dependant computations for roughness"""
    cdef double cose, sine, cosi, sini, phir, f

    cose = cos(theta)
    sine = sin(theta)
    cosi = cos(theta0)
    sini = sin(theta0)

    phir = phi * pi / 180.  # radians

    if phi == 180:
        f = 0
    else:
        f = exp( -2 * tan(phir/2))

    return theta0, theta, cose, sine,cosi,sini,phir,f


cdef T3Double compute_roughness(T8Double geom_infos,double R) nogil:
    """phi in degrees for numerical stability"""
    cdef double xidz, cose,sine, cosi, sini, e2R, e1R, e2R0, e1R0, mu_b, mu0_b, phir, f, mu0_e, mu_e, S, tR

    theta0,theta ,cose,sine ,cosi,sini,phir,f = geom_infos

    xidz = 1 / sqrt(1 + pi * (tan(R) ** 2))
    e1R, e2R, e1R0, e2R0, tR = fast_e1_e2(R, theta, theta0)

    mu_b = xidz * (cose + sine * tR * e2R / (2 - e1R))
    mu0_b = xidz * (cosi + sini * tR * e2R0 / (2 - e1R0))

    if theta0 <= theta:
        mu0_e = xidz * (cosi + sini * tR * (cos(phir) * e2R + (sin(phir / 2) ** 2) * e2R0) / (
                2 - e1R - (phir/ pi) * e1R0))
        mu_e = xidz * (cose + sine * tR * (e2R - (sin(phir / 2) ** 2) * e2R0) / (
                2 - e1R - (phir / pi) * e1R0))

        S = mu_e * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cosi / mu0_b)
    else:
        mu0_e = xidz * (cosi + sini * tR * (e2R0 - sin(phir / 2) ** 2 * e2R) / (
                2 - e1R0 - (phir / pi) * e1R))

        mu_e = xidz * (cose + sine * tR * (cos(phir) * e2R0 + sin(phir / 2) ** 2 * e2R) / (
                2 - e1R0 - (phir / pi) * e1R))

        S = mu_e * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cose / mu_b)

    return mu0_e,mu_e, S


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Hapke_vect(double[:] SZA, double[:] VZA, double[:] DPHI, double[:] W, double[:] R1, double[:] BB,
double[:] CC, double[:] HH, double[:] B0):
    """ Returns a matrix of reflectance of shape Nx , D """

    cdef double theta0r, thetar, phir, r, c, b, h, b0, w, reff, MUP, MU, S,  ctheta, theta
    cdef double bc,b2,P, B, H, H0, gamma, alpha# var temporaires
    cdef T8Double geom_infos

    cdef Py_ssize_t Nx = W.shape[0]
    cdef Py_ssize_t D = SZA.shape[0]

    REFF = np.zeros((Nx,D),dtype=DTYPE)
    cdef double[:,:] REFF_view = REFF  #memoryview
    cdef Py_ssize_t n,d # loop indices

    for d in prange(D, nogil=True):
        theta0r = SZA[d] * pi / 180
        thetar = VZA[d] * pi / 180
        geom_infos = _geom_roughness(theta0r,thetar,DPHI[d])

        phir = DPHI[d] * pi / 180
        ctheta = cos(theta0r) * cos(thetar) + sin(thetar) * sin(theta0r) * cos(phir)
        theta = acos(ctheta)

        alpha = 4 * cos(theta0r)

        for n in range(Nx):
            r = R1[n] * pi / 180  # radians
            MUP, MU, S = compute_roughness(geom_infos,r)
            c = CC[n]
            b = BB[n]
            h = HH[n]
            b0 = B0[n]
            w = W[n]
            bc = 2 * b * ctheta
            b2 = b**2
            P =  (1 - c) * (1 - b2) / ((1 + bc + b2)**1.5)
            P = P + c *(1 - b2) / ((1- bc + b2)**1.5)

            B = b0 * h / ( h + tan(theta/2) )
            gamma = sqrt(1 - w)
            if VARIANT == 1993:
                H0 = H_1993(MUP,gamma)
                H = H_1993(MU,gamma)
            else:
                H0 = H_2002(MUP,gamma)
                H = H_2002(MU,gamma)
            reff = ( (w / (MU + MUP)) * ((1 + B) * P + (H0 * H) - 1)  )
            reff = reff *  S * MUP  / alpha
            REFF_view[n,d] = reff
    return REFF
