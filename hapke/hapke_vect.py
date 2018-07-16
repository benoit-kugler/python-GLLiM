"""Python implementation (for performances) of Hapke's model.
Port from Matlab version from S Doute
"""
import numpy as np

# INPUTS %
# SZA : solar zenith angle in degrees
# VZA : view zenith angle in degrees
# phi: azimuth in degrees
# W : single scattering albedo
# R : macroscopic roughness
# BB : assymetry of the phase function
# CC : fraction of the backward scattering
# HH : angular width of the opposition effect
# B0 : amplitude of the opposition effect

# OUTPUT %
# REFF  : bidirectional reflectance factor


def e1(R, x):
    y = np.zeros(R.shape)
    arg = (np.tan(R) * np.tan(x) * np.pi)
    mask =  ~ (np.isclose(R,0) + np.isclose(x,0))
    y[mask] = np.exp(-2/ arg[mask])
    return y

def e2(R, x):
    y = np.zeros(R.shape)
    arg = (np.tan(R)**2 * np.tan(x)**2 * np.pi )
    mask = ~ (np.isclose(R,0) + np.isclose(x,0))
    y[mask] = np.exp(-1 / arg[mask])
    return y

def roughness(theta0,theta,phi, R):
    xidz = 1 / np.sqrt(1 + np.pi * (np.tan(R) ** 2))
    cose = np.cos(theta)
    sine = np.sin(theta)
    cosi = np.cos(theta0)
    sini = np.sin(theta0)
    mu_b = xidz * (cose + sine * np.tan(R) * e2(R,theta) / (2 - e1(R,theta)))
    mu0_b = xidz * (cosi + sini * np.tan(R) * e2(R,theta0) / (2 - e1(R,theta0)))

    f = np.exp( -2 * np.tan(phi/2))


    mask1 = theta0 <= theta
    mask2 = ~ mask1
    mu0_e , mu_e, S = np.empty(theta.shape) ,np.empty(theta.shape), np.empty(theta.shape)

    # Case 1
    mu0_e1 = xidz * (cosi + sini * np.tan(R) * (np.cos(phi) * e2(R,theta) + (np.sin(phi / 2) ** 2) * e2(R,theta0)) / (
                2 - e1(R,theta) - (phi/ np.pi) * e1(R,theta0)))
    mu_e1 = xidz * (cose + sine * np.tan(R) * (e2(R,theta) - (np.sin(phi / 2) ** 2) * e2(R,theta0)) / (
                2 - e1(R,theta) - (phi / np.pi) * e1(R,theta0)))

    S1 = mu_e1 * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cosi / mu0_b)

    #Case 2
    mu0_e2 = xidz * (cosi + sini * np.tan(R) * (e2(R,theta0) - np.sin(phi / 2) ** 2 * e2(R,theta)) / (
                2 - e1(R,theta0) - (phi / np.pi) * e1(R,theta)))
    mu_e2 = xidz * (cose + sine * np.tan(R) * (np.cos(phi) * e2(R,theta0) + np.sin(phi / 2) ** 2 * e2(R,theta)) / (
                2 - e1(R,theta0) - (phi / np.pi) * e1(R,theta)))

    S2 = mu_e2 * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cose / mu_b)


    mu0_e[mask1] = mu0_e1[mask1]
    mu0_e[mask2] = mu0_e2[mask2]
    mu_e[mask1] = mu_e1[mask1]
    mu_e[mask2] = mu_e2[mask2]
    S[mask1] = S1[mask1]
    S[mask2] = S2[mask2]
    return mu0_e,mu_e, S

def H_2002(x,y):
    return (1 + 2 * x) / (1 + 2 * x * y)


def H_1993(x,y): # eq 3 from icarus Schimdt
    u = (1 - y) / (1 + y)
    t = 1 - (1-y) * x * (  u  +  (1 - (0.5 + x) * u) * np.log((1+x)/x)    )
    return 1 / t

def Hapke_vect (SZA, VZA, DPHI, W, R, BB, CC, HH, B0, variant="2002"):
    # En radian
    theta0r = SZA * np.pi/ 180
    thetar = VZA * np.pi/ 180
    phir = DPHI * np.pi/ 180
    R = R * np.pi/ 180
    CTHETA = np.cos(theta0r) * np.cos(thetar) + np.sin(thetar) * np.sin(theta0r) * np.cos(phir)
    THETA = np.arccos(CTHETA)


    MUP, MU, S = roughness(theta0r,thetar,phir,R)


    P = (1 - CC) * (1 - BB**2) / ((1 + 2 * BB * CTHETA + BB**2)**(3/2))
    P = P + CC *(1 - BB**2) / ((1- 2 * BB * CTHETA + BB**2)**(3/2))

    B = B0 * HH / ( HH + np.tan(THETA/2) )

    gamma = np.sqrt(1 - W)
    # H0 = (1 + 2 * MUP) / (1 + 2 * MUP * gamma)
    # H = (1 + 2 * MU) / (1 + 2 * MU * gamma)
    if variant == "1993":
        H0 = H_1993(MUP,gamma)
        H = H_1993(MU,gamma)
    else:
        H0 = H_2002(MUP,gamma)
        H = H_2002(MU,gamma)

    REFF = W / 4 / (MU + MUP) * ((1 + B) * P + H0 * H - 1)

    REFF = S * REFF * MUP / np.cos(theta0r)
    return REFF










