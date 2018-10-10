"""Implements MCMC-GLLiM hybrid approach (is-GLLiM; FF)"""
import logging
import time
from typing import Callable

import coloredlogs
import matplotlib
import numpy
import numpy as np

matplotlib.use("QT5Agg")
from matplotlib import pyplot

from Core.gllim import GLLiM, jGLLiM
from Core.probas_helper import densite_melange, chol_loggauspdf_diag, chol_loggausspdf
from tools import context
from tools.experience import Experience


def gllim_q(Xs: numpy.ndarray, Y: numpy.ndarray, gllim: GLLiM):
    """Compute conditionnal density X | Y = y
    Y shape : Ny,D
    Xs shape : Ny, Nsample, L
    return shape Ny, Nsample
    """
    Ny, Nsample, _ = Xs.shape
    meanss, weightss, _ = gllim._helper_forward_conditionnal_density(Y)
    covs = gllim.SigmakListS
    out = np.array([densite_melange(X, weights, means, covs) for means, weights, X in zip(meanss, weightss, Xs)])
    return out


def p_tilde(FXs, Y, noise_cov, noise_mean):
    """Compute non normalized likelihood.
    Y shape : Ny,D
    FXs shape : Ny, Nsample, D
    noise_cov : shape D (diagonal cov) or shape D,D (full)
    noise_mean : shape D (offfset)
    return shape Ny, Nsample
    """
    Ny, Nsample, _ = Xs.shape
    out = numpy.empty((Ny, Nsample))
    if noise_cov.ndim == 1:
        for i, (y, X) in enumerate(zip(Y, Xs)):
            out[i] = chol_loggauspdf_diag(F(X).T + noise_mean.T[:, None], y[:, None], noise_cov)
    else:
        chol = np.linalg.cholesky(noise_cov)
        for i, (y, X) in enumerate(zip(Y, Xs)):
            fx = F(X)
            if np.isfinite(fx).all():
                out[i] = chol_loggausspdf(F(X).T + noise_mean.T[:, None], y[:, None], _, cholesky=chol)
            else:
                out[i] = np.NAN * np.ones(Nsample)
    return numpy.exp(out)  # we used log pdf so far


def mean_IS(Y, gllim, F, r, Nsample=50000):
    G = lambda x: x
    return compute_is(Y, gllim, G, F, r, Nsample=Nsample)


def _clean_integrate(G: Callable[[np.ndarray], np.ndarray], Xs, ws):
    """G : fonction to integrate
        Xs : samplings
        ws : weights
    """
    GX = G(Xs)
    mask = ~ np.array([(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in Xs])
    mask1 = numpy.isfinite(GX).prod(axis=2)
    if GX.ndim == 4:
        mask1 = mask1.prod(axis=2)
    mask1 = ~ numpy.asarray(mask1, dtype=bool)
    mask2 = ~ numpy.isfinite(ws)
    Nsample = ws.shape[1]
    logging.debug(f"Average ratio of G-non-compatible samplings : {mask1.sum(axis=1).mean() / Nsample:.5f}")
    logging.debug(f"Average ratio of infinite weights : {mask2.sum(axis=1).mean() / Nsample:.5f}")
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / Nsample:.5f}")
    mask = mask | mask1 | mask2
    ws[mask] = 0
    if GX.ndim == 4:
        GX[mask, :, :] = 0
        ws = ws[:, :, None, None]
    else:
        GX[mask, :] = 0
        ws = ws[:, :, None]
    return numpy.sum(GX * ws, axis=1) / numpy.sum(ws, axis=1)


def _weigth_sample(gllim, Y, F, noise_cov, noise_mean, Nsample):
    Xs = gllim.predict_sample(Y, nb_per_Y=Nsample)
    ws = p_tilde(Xs, Y, F, noise_cov, noise_mean) / gllim_q(Xs, Y, gllim)
    return Xs, ws


def compute_is(Y, gllim, G, F, noise_cov, noise_mean, Nsample=50000):
    """Compute E[ G(X) | Y = y] for given parameters (gllim) and noise (r)
    G(X) has to be vectoriel (for generality), ie G : shape (Ny, Nsample, _) -> shape (Ny, Nsample,_).
    Return shape : (Ny, _)
    """
    ti = time.time()
    Xs, ws = _weigth_sample(gllim, Y, F, noise_cov, noise_mean, Nsample)
    logging.debug(f"Samplings and weights computed in {time.time()-ti:.3f} s")
    return _clean_integrate(G, Xs, ws)



def _test():
    c: context.abstractExpFunction = context.InjectiveFunction(1)()

    exp, gllim = Experience.setup(context.InjectiveFunction(1), 30, partiel=None, with_plot=True,
                                  regenere_data=False, with_noise=50, N=10000, method="sobol",
                                  mode="l", init_local=100,
                                  sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    X = c.get_X_sampling(1)
    Y = c.F(X)
    Xs = c.get_X_sampling(200)[None, :]
    dens = p_tilde(Xs, Y, c.F, 50)[0]
    dens2 = gllim_q(Xs, Y, gllim)[0]
    pyplot.scatter(Xs[0, :, 0], np.log(dens), label="p_tilde")
    pyplot.scatter(Xs[0, :, 0], np.log(dens2), label="q")
    pyplot.axvline(X[0])
    ws = p_tilde(Xs, Y, c.F, 50) / gllim_q(Xs, Y, gllim)
    ws[~ numpy.isfinite(ws)] = 0
    pyplot.scatter(Xs[0, :, 0], np.log(ws), label="w_i")
    pyplot.legend()
    pyplot.show()





if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # _test()
