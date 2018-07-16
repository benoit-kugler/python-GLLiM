"""Implemente le modèle de Hapke en calcul symbolique.
Les paramètres géométriques sont importés et évalués. Les autres paramètres sont des symbols.
"""
import time

import dill
import numpy as np
import sympy as sp
from sympy.printing.theanocode import theano_function

DEFAULT_SAVEPATH = "__dF.dill"

def import_geometries(context):
    geom = context.geometries
    theta0 , theta , phi = geom[:,0]
    # En radian
    theta0r = theta0 * np.pi/ 180.
    thetar = theta * np.pi/ 180.
    phir = phi * np.pi/ 180.
    CTHETA = np.cos(theta0r) * np.cos(thetar) + np.sin(thetar) * np.sin(theta0r) * np.cos(phir)
    THETA = np.arccos(CTHETA)
    return theta0r, thetar, phi, CTHETA,THETA


w, theta_b, b, c, HH, B0 = sp.symbols("w theta_b b c H B_0")
theta_br = theta_b * sp.pi / 180.

def Fi(i,e,dphi,ctheta,theta) -> sp.Mul:
    """En radians, sauf dphi"""
    MUP, MU, S = roughness(i,e,dphi)

    P = (1 - c) * (1 - b**2) / ((1 + 2 * b * ctheta + b**2)**(3/2))
    P = P + c *(1 - b**2) / ((1 - 2 * b * ctheta + b**2)**(3/2))

    B = B0 * HH / ( HH + np.tan(theta/2) )

    gamma = sp.sqrt(1 - w)
    H0 = (1 + 2 * MUP) / (1 + 2 * MUP * gamma)
    H = (1 + 2 * MU) / (1 + 2 * MU * gamma)

    BRDF = w / 4 / (MU + MUP) * ((1 + B) * P + H0 * H - 1)

    BRDF = S * BRDF * MUP / np.cos(i)
    return BRDF

def e1(x):
    tx = np.tan(x)
    if tx == 0:
        print("0")
        return sp.Integer(0)
    den = sp.tan(theta_br) *  tx * sp.pi
    return sp.exp(- 2 / den)


def e2(x):
    tx = np.tan(x)
    if tx == 0:
        print("0")
        return sp.Integer(0)
    den =  sp.tan(theta_br)**2 * tx**2 * sp.pi
    return sp.exp(-1 / den)


def roughness(theta0,theta,phi):
    """Prend une géométrie (phi en degrees pour des raisons numériques)
     et renvoie des EXPRESSIONS  en theta_b """
    xidz = 1. / sp.sqrt(1. + sp.pi * (sp.tan(theta_br) ** 2))
    phir = phi * np.pi / 180.  # radians
    cose = np.cos(theta)
    sine = np.sin(theta)
    cosi = np.cos(theta0)
    sini = np.sin(theta0)
    mu_b = xidz * (cose + sine * sp.tan(theta_br) * e2(theta) / (2 - e1(theta)))
    mu0_b = xidz * (cosi + sini * sp.tan(theta_br) * e2(theta0) / (2 - e1(theta0)))

    if phi == 180: # tan infinity
        print("f0")
        f = sp.Integer(0)
    else:
        f = np.exp( -2 * np.tan(phir/2))

    if theta0 <= theta:
        print("cas1")
        mu0_e = xidz * (cosi + sini * sp.tan(theta_br) * (np.cos(phir) * e2(theta) + (np.sin(phir / 2) ** 2) * e2(theta0)) / (
                    2 - e1(theta) - (phir/ sp.pi) * e1(theta0)))
        mu_e = xidz * (cose + sine * sp.tan(theta_br) * (e2(theta) - (np.sin(phir / 2) ** 2) * e2(theta0)) / (
                    2 - e1(theta) - (phir / sp.pi) * e1(theta0)))

        S = mu_e * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cosi / mu0_b)
    else:
        print("cas2")
        mu0_e = xidz * (cosi + sini * sp.tan(theta_br) * (e2(theta0) - np.sin(phir / 2) ** 2 * e2(theta)) / (
                    2 - e1(theta0) - (phir / sp.pi) * e1(theta)))
        mu_e = xidz * (cose + sine * sp.tan(theta_br) * (np.cos(phir) * e2(theta0) + np.sin(phir / 2) ** 2 * e2(theta)) / (
                    2 - e1(theta0) - (phir / sp.pi) * e1(theta)))

        S = mu_e * cosi * xidz / mu_b / mu0_b / (1 - f + f * xidz * cose / mu_b)

    return mu0_e,mu_e, S


def lambdify_F(context):
    F = sp.Matrix([Fi(*t) for t in zip(*import_geometries(context))])
    return sp.lambdify([w, theta_b, b, c, HH, B0],F,"np")

# def lambdify_dF(context):
#     print("Computing F and dF expression...")
#     ti = time.time()
#     F = sp.Matrix([Fi(*t) for t in zip(*import_geometries(context))])
#     dw = sp.diff(F,w)
#     dtheta_b = sp.diff(F,theta_b)
#     db = sp.diff(F,b)
#     dc = sp.diff(F,c)
#     dHH = sp.diff(F,HH)
#     dB0 = sp.diff(F,B0)
#     diff = sp.Matrix([[*dw],[*dtheta_b],[*db],[*dc],[*dHH],[*dB0]]).T
#     print("Done in {:.3f} s".format(time.time() - ti))
#     print("Computing dF function (not vectorized)...")
#     ti = time.time()
#     dF_func = sp.lambdify([w, theta_b, b, c, HH, B0],diff)
#
#     inputs = [w, theta_b, b, c, HH, B0]
#     # dF_func = theano_function(inputs,[diff],{x:'float64' for x in inputs})
#
#     print("Done in {:.3f} s".format(time.time() - ti))
#     return dF_func

def lambdify_dF(context):
    print("Computing dF one geometrie at a time...")
    inputs = [w, theta_b, b, c, HH, B0]
    components = []
    for i , t in enumerate( zip(*import_geometries(context)) ):
        print("Component {} ...".format(i+1))
        if not i == 7:
            continue
        ti = time.time()
        F = Fi(*t)
        dw = sp.diff(F,w)
        dtheta_b = sp.diff(F,theta_b)
        db = sp.diff(F,b)
        dc = sp.diff(F,c)
        dHH = sp.diff(F,HH)
        dB0 = sp.diff(F,B0)
        diff = sp.Matrix([[dw,dtheta_b,db,dc,dHH,dB0]])
        print("\tF and dF expression computed in {:.3f} s".format(time.time() - ti))
        print(diff.shape)
        print("\tComputing dF function with Theano...")
        ti = time.time()

        dF_func = theano_function(inputs,[dw],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        dF_func = theano_function(inputs,[dtheta_b],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        dF_func = theano_function(inputs,[db],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        dF_func = theano_function(inputs,[dc],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        dF_func = theano_function(inputs,[dHH],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        dF_func = theano_function(inputs,[dB0],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        print("ok")
        dF_func = theano_function(inputs,[dw,dtheta_b,db,dc,dHH,dB0],{x:'float64' for x in inputs}, dims={x:1 for x in inputs})
        components.append(dF_func)
        print("\tDone in {:.3f} s".format(time.time() - ti))

    def dF(X):
        DFs = np.empty((X.shape[0],len(components),X.shape[1]))   # N , D , L
        for i, df in enumerate(components):
            DFs[:,i,:] = df(*X.T)[0].T
        return DFs

    return dF


def calcule_rang(context):
    _ , numpudiff , _ = lambdify_dF(context)
    N = 200000
    X = context.get_X_sampling(N)
    print("Evaluations des différentielles...")
    diffs = numpudiff(*X.T).transpose((2,0,1))
    print("Calcul des rangs...")
    rangs = np.linalg.matrix_rank(diffs)
    r = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
    for ra in rangs:
        r[ra] += 1
    print("Nombre de points testés {} , rangs : {}".format(N,r))



def save_dF(context,savepath=DEFAULT_SAVEPATH):
    """Uses context geometries to compute dF and pickle it"""
    l_dF = lambdify_dF(context)
    dill.settings["recurse"] = True
    with open(savepath,'wb') as file:
        dill.dump(l_dF,file)

def load_dF(savepath=DEFAULT_SAVEPATH):
    with open(savepath, 'rb') as file:
        dF = dill.load(file)
    return dF









