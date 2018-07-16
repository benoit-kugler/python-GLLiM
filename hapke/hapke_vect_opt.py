"""Python implementation (for performances) of Hapke's model.
Port from Matlab version from S Doute
"""
import numexpr as ne
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
    pi = np.pi
    u = ne.evaluate("exp(-2 / (tan(R) * tan(x) * pi))")
    return u


def e2(R, x):
    pi = np.pi
    u =  ne.evaluate("exp(-1 / (tan(R)**2 * tan(x)**2 * pi ))")
    return u

def roughness(theta0,theta,phi, R):
    """phi in degrees for numerical stability"""
    pi = np.pi
    xidz = ne.evaluate("1 / sqrt(1 + pi * (tan(R) ** 2))")
    cose = np.cos(theta)
    sine = np.sin(theta)
    cosi = np.cos(theta0)
    sini = np.sin(theta0)
    e2R = e2(R,theta)
    e1R = e1(R,theta)
    e2R0 = e2(R,theta0)
    e1R0 = e1(R,theta0)
    mu_b = ne.evaluate("xidz * (cose + sine * tan(R) * e2R / (2 - e1R))")
    mu0_b = ne.evaluate("xidz * (cosi + sini * tan(R) * e2R0 / (2 - e1R0))")

    phir = phi * np.pi / 180.  # radians
    f = np.zeros(phi.shape)
    mask = phi == 180
    f[~ mask] = ne.evaluate("exp( -2 * tan(phir/2))")[~ mask]



    mask1 = theta0 <= theta
    mask2 = ~ mask1
    mu0_e , mu_e, S = np.empty(theta.shape) ,np.empty(theta.shape), np.empty(theta.shape)

    # Case 1
    mu0_e1 = ne.evaluate("""xidz * (cosi + sini * tan(R) * (cos(phir) * e2R + (sin(phir / 2) ** 2) * e2R0) / (
                2 - e1R - (phir/ pi) * e1R0))""")
    mu_e1 = ne.evaluate("""xidz * (cose + sine * tan(R) * (e2R - (sin(phir / 2) ** 2) * e2R0) / (
                2 - e1R - (phir / pi) * e1R0))""")

    S1 = ne.evaluate("mu_e1 * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cosi / mu0_b)")

    #Case 2
    mu0_e2 = ne.evaluate("""xidz * (cosi + sini * tan(R) * (e2R0 - sin(phir / 2) ** 2 * e2R) / (
                2 - e1R0 - (phir / pi) * e1R))""")

    mu_e2 = ne.evaluate("""xidz * (cose + sine * tan(R) * (cos(phir) * e2R0 + sin(phir / 2) ** 2 * e2R) / (
                2 - e1R0 - (phir / pi) * e1R))""")

    S2 = ne.evaluate("""mu_e2 * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cose / mu_b)""")


    mu0_e[mask1] = mu0_e1[mask1]
    mu0_e[mask2] = mu0_e2[mask2]
    mu_e[mask1] = mu_e1[mask1]
    mu_e[mask2] = mu_e2[mask2]
    S[mask1] = S1[mask1]
    S[mask2] = S2[mask2]
    return mu0_e,mu_e, S

def H_2002(x,y):
    return ne.evaluate("""(1 + 2 * x) / (1 + 2 * x * y)""")


def H_1993(x,y): # eq 3 from icarus Schimdt
    u = (1 - y) / (1 + y)
    return ne.evaluate("""1 / ( 1 - (1-y) * x * (  u  +  (1 - (0.5 + x) * u) * log((1+x)/x) ))""")


def Hapke_vect (SZA, VZA, DPHI, W, R, BB, CC, HH, B0, variant="2002"):
    # En radian
    theta0r = SZA * np.pi/ 180
    thetar = VZA * np.pi/ 180
    phir = DPHI * np.pi/ 180
    R = R * np.pi/ 180
    CTHETA = ne.evaluate("""cos(theta0r) * cos(thetar) + sin(thetar) * sin(theta0r) * cos(phir)""")
    THETA = np.arccos(CTHETA)


    MUP, MU, S = roughness(theta0r,thetar,DPHI,R)


    P =  ne.evaluate("(1 - CC) * (1 - BB**2) / ((1 + 2 * BB * CTHETA + BB**2)**(3/2))")
    P = ne.evaluate("P + CC *(1 - BB**2) / ((1- 2 * BB * CTHETA + BB**2)**(3/2))")

    B = ne.evaluate("B0 * HH / ( HH + tan(THETA/2) )")

    gamma = np.sqrt(1 - W)
    # H0 = (1 + 2 * MUP) / (1 + 2 * MUP * gamma)
    # H = (1 + 2 * MU) / (1 + 2 * MU * gamma)
    if variant == "1993":
        H0 = H_1993(MUP,gamma)
        H = H_1993(MU,gamma)
    else:
        H0 = H_2002(MUP,gamma)
        H = H_2002(MU,gamma)

    REFF = ne.evaluate("W / 4 / (MU + MUP) * ((1 + B) * P + H0 * H - 1)")

    REFF = ne.evaluate("S * REFF * MUP / cos(theta0r)")
    return REFF






