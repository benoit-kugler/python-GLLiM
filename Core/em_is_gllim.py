"""Implements a crossed EM - GLLiM algorith to evaluate noise in observations."""
import logging

import numpy as np

from Core.gllim import jGLLiM
from experiences.importance_sampling import compute_is
from tools import context

Ntrain = 1000
init_X_precision_factor = 10
N_sample_IS = 5000

INIT_PREC = 50

maxIter = 5
maxIterGlliM = 3
K = 20


def _G(Xs, Y, F):
    """Fonctionnelle à intégrer
    X shape : Ny,N,L
    Y shape : Ny,D
    return shape (Ny,N,1)
    """
    for X, y in zip(Xs, Y):
        fx = F(X)
        yield np.square((fx - y[None, :]) / fx).sum(axis=1)[:, None]  # need a 3D array


gllim = jGLLiM(K, sigma_type="full")


def _e_step(context, Yobs, current_prec, current_ck):
    Xtrain, Ytrain = context.get_data_training(Ntrain)
    Ytrain = context.add_noise_data(Ytrain, precision=current_prec)

    rho = np.ones(gllim.K) / gllim.K
    m = current_ck
    precisions = init_X_precision_factor * np.array([np.eye(Xtrain.shape[1])] * gllim.K)
    rnk = gllim._T_GMM_init(Xtrain, 'random',
                            weights_init=rho, means_init=m, precisions_init=precisions)
    assert np.isfinite(rnk).all()
    gllim.fit(Xtrain, Ytrain, {"rnk": rnk}, maxIter=maxIterGlliM)
    gllim.inversion()
    G = lambda X: np.array(list(_G(X, Yobs, context.F)))
    esp = compute_is(Yobs, gllim, G, context.F, current_prec, Nsample=N_sample_IS)[:, 0]  # scalar value
    return esp, gllim.ckList


def _m_step(Yobs, estimated_esp):
    D = Yobs.shape[1]
    return D / np.sum(estimated_esp)


def run_em_is_gllim(Yobs, contex: context.abstractHapkeModel):
    current_prec = INIT_PREC
    current_ck = contex.get_X_uniform(K)
    for current_iter in range(maxIter):
        estimated_esp, current_ck = _e_step(contex, Yobs, current_prec, current_ck)
        print(estimated_esp)
        return
        current_prec = _m_step(Yobs, estimated_esp)
        logging.debug(f"New estimated noise : {current_prec:.4f}")
    return current_prec
