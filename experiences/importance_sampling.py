"""Implements MCMC-GLLiM hybrid approach (is-GLLiM; FF)"""
import logging
import time

import coloredlogs
import matplotlib
import numpy
import numpy as np

matplotlib.use("QT5Agg")
from matplotlib import pyplot

from Core.gllim import GLLiM, jGLLiM
from Core.log_gauss_densities import densite_melange, chol_loggauspdf_diag
from tools import context
from tools.experience import Experience


def q(Xs: numpy.ndarray, Y: numpy.ndarray, gllim: GLLiM):
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


def p_tilde(Xs, Y, F, r):
    """Compute non normalized ll.
    Y shape : Ny,D
    Xs shape : Ny, Nsample, L
    F : callable shape(Nx,L) -> shape(Nx,D)
    r : Covariances of gaussian noise is given by (diag(y/r))^2
    return shape Ny, Nsample
    """
    Ny, Nsample, _ = Xs.shape
    out = numpy.empty((Ny, Nsample))
    for i, (y, X) in enumerate(zip(Y, Xs)):
        cov = numpy.square(y / r)
        out[i] = chol_loggauspdf_diag(F(X).T, y[:, None], cov[:, None])
    return numpy.exp(out)  # we used log pdf so far


def mean_IS(Y, gllim, F, r, Nsample=50000):
    G = lambda x: x
    return compute_is(Y, gllim, G, F, r, Nsample=Nsample)


def compute_is(Y, gllim, G, F, r, Nsample=50000):
    """Compute E[ G(X) | Y = y] for given parameters (gllim) and noise (r)
    G(X) has to be vectoriel (for generality), ie G : shape (Ny, Nsample, _) -> shape (Ny, Nsample,_).
    Return shape : (Ny, _)
    """
    Xs = gllim.predict_sample(Y, nb_per_Y=Nsample)
    mask = ~ np.array([(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in Xs])
    ti = time.time()
    ws = p_tilde(Xs, Y, F, r) / q(Xs, Y, gllim)
    logging.debug(f"Sampling weights computed in {time.time()-ti:.3f} s")
    GX = G(Xs)
    mask1 = ~ numpy.asarray(numpy.isfinite(GX).prod(axis=2), dtype=bool)
    mask2 = ~ numpy.isfinite(ws)
    logging.debug(f"Average ratio of G-non-compatible samplings : {mask1.sum(axis=1).mean() / Nsample:.5f}")
    logging.debug(f"Average ratio of infinite weights : {mask2.sum(axis=1).mean() / Nsample:.5f}")
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / Nsample:.5f}")
    mask = mask | mask1 | mask2
    ws[mask] = 0
    GX[mask, :] = 0
    return numpy.sum(GX * ws[:, :, None], axis=1) / numpy.sum(ws, axis=1, keepdims=True)




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
    dens2 = q(Xs, Y, gllim)[0]
    pyplot.scatter(Xs[0, :, 0], np.log(dens), label="p_tilde")
    pyplot.scatter(Xs[0, :, 0], np.log(dens2), label="q")
    pyplot.axvline(X[0])
    ws = p_tilde(Xs, Y, c.F, 50) / q(Xs, Y, gllim)
    ws[~ numpy.isfinite(ws)] = 0
    pyplot.scatter(Xs[0, :, 0], np.log(ws), label="w_i")
    pyplot.legend()
    pyplot.show()





if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # _test()
