"""Implements a crossed EM - GLLiM algorith to evaluate noise in observations."""
import logging

import coloredlogs
import numpy as np

from Core.gllim import jGLLiM
from experiences.importance_sampling import compute_is
from tools import context

# GLLiM parameters
Ntrain = 50000
K = 30
init_X_precision_factor = 10
maxIterGlliM = 5


N_sample_IS = 100000

INIT_PREC = 50  # initial noise inverse
maxIter = 5


def _G(Xs, Y, F):
    """Fonctionnelle à intégrer
    X shape : Ny,N,L
    Y shape : Ny,D
    return shape (Ny,N,1)
    """
    for X, y in zip(Xs, Y):
        fx = F(X)
        yield np.square(fx - y[None, :] / fx).sum(axis=1)[:, None]  # need a 3D array


gllim = jGLLiM(K, sigma_type="full")


def _e_step(context, Yobs, current_prec_2, current_ck):
    current_prec = np.sqrt(current_prec_2)
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
    F = lambda X: context.F(X, check=False)
    G = lambda X: np.array(list(_G(X, Yobs, F)))
    esp = compute_is(Yobs, gllim, G, F, current_prec, Nsample=N_sample_IS)[:, 0]  # scalar value
    return esp, gllim.ckList


def _m_step(Yobs, estimated_esp):
    D = Yobs.shape[1]
    N = estimated_esp.shape[0]
    return D * N / np.sum(estimated_esp)


def run_em_is_gllim(Yobs, contex: context.abstractHapkeModel):
    logging.info(f"Starting EM-iS for noise (inital precision : {INIT_PREC})")
    cur_prec_2 = INIT_PREC ** 2
    current_ck = contex.get_X_uniform(K)
    for current_iter in range(maxIter):
        estimated_esp, current_ck = _e_step(contex, Yobs, cur_prec_2, current_ck)
        cur_prec_2 = _m_step(Yobs, estimated_esp)
        logging.info(f"New estimated PRECISION : {np.sqrt(cur_prec_2):.4f}")
    return cur_prec_2


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    cont = context.LabContextOlivine(partiel=(0, 1, 2, 3))
    # Yobs = cont.get_observations()
    _, Yobs = cont.get_data_training(1000)
    Yobs = cont.add_noise_data(Yobs, 30)
    run_em_is_gllim(Yobs, cont)
