import json
import time

import numpy as np

from gllim import GLLiM
from old.gllim_backup import OldGLLiM
from tests import show_diff


def is_egal(modele1,modele2,verbose=False):
    if verbose:
        show_diff(modele1[0], modele2[0],'Diff pi')
        show_diff(modele1[1], modele2[1],'Diff c')
        show_diff(modele1[2], modele2[2],'Diff Gamma')
        show_diff(modele1[3], modele2[3],'Diff A')
        show_diff(modele1[4], modele2[4],'Diff b')
        show_diff(modele1[5], modele2[5],'Diff Sigma')

    assert np.allclose(modele1[0], modele2[0])
    assert np.allclose(modele1[1], modele2[1])
    assert np.allclose(modele1[2], modele2[2])
    assert np.allclose(modele1[3], modele2[3])
    assert np.allclose(modele1[4], modele2[4])
    assert np.allclose(modele1[5], modele2[5])



def _test_next_theta(g1, g2, g3, T, Y):
    print("Testing compute_next_theta ")
    ti = time.time()
    theta1 = g1.compute_next_theta(T, Y)
    print(f"\tCython sequentiel : {time.time() - ti:.3f} s")

    ti = time.time()
    theta2 = g2.compute_next_theta(T, Y)
    print(f"\tCython parallel   : {time.time() - ti:.3f} s")

    ti = time.time()
    theta3 = g3.compute_next_theta(T, Y)
    print(f"\tPython (old)      : {time.time() - ti:.3f} s \n")


    assert np.isclose(np.sum(theta1[0]), 1)
    assert np.isclose(np.sum(theta2[0]), 1)
    assert np.isclose(np.sum(theta3[0]), 1)
    is_egal(theta1, theta3)
    is_egal(theta1, theta2)


def _test_rnk(g1, g2, g3, T, Y):
    print("Testing compute_rnk ")
    ti = time.time()
    ll3, log_rnk3 = g3._compute_rnk(T, Y)
    rnk3 = np.exp(log_rnk3)
    ll3 = ll3[:,0]
    print(f"\tPython (old)      : {time.time() - ti:.3f} s ")

    ti = time.time()
    rnk1, ll1 = g1._compute_rnk(T, Y)
    print(f"\tCython sequentiel : {time.time() - ti:.3f} s")


    assert np.allclose(rnk1, rnk3)
    assert np.allclose(ll1, ll3)



def compare_para(N=1000, D=10, Lt=4, Lw=0, K=4):
    """Compare cythons implementations : sequential vs parallel"""
    Y = np.random.random_sample((N, D)) + 2
    T = np.random.random_sample((N, Lt))

    for gamma_type in ("full", "diag", "iso"):
        for sigma_type in ("full", "diag", "iso"):
            g1 = GLLiM(K, Lw, sigma_type=sigma_type, gamma_type=gamma_type, parallel=False)
            g2 = GLLiM(K, Lw, sigma_type=sigma_type, gamma_type=gamma_type, parallel=True)
            g3 = OldGLLiM(K, Lw, sigma_type=sigma_type, gamma_type=gamma_type)

            print(f"\nGamma type: {g1.gamma_type}, Sigma type : {g1.sigma_type}")
            rnk = np.random.random_sample((T.shape[0], g1.K))
            rnk /= np.sum(rnk, axis=1, keepdims=True)
            init = {"rnk": rnk}
            init = None

            # with open("tmp.json","r") as f:
            #     init = json.load(f)

            g3.init_fit(T, Y, init)
            g1.init_fit(T, Y, init)
            g2.init_fit(T, Y, init)

            _test_rnk(g1, g2, g3, T, Y)
            # _test_next_theta(g1, g2, g3, T, Y)


def compare_complet(N,D):
    compare_para(N,D,3,0,K=40)
    # compare_para(N,D,4,1,K=40)
    # compare_para(N,D,0,4,K=40)


if __name__ == '__main__':
    compare_complet(10000,3)