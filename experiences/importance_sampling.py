"""Implements MCMC-GLLiM hybrid approach (is-GLLiM; FF)"""
import logging

import coloredlogs
import numpy
import numpy as np
import matplotlib

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
    out = numpy.empty((Ny, Nsample))
    for i, (means, weights, X) in enumerate(zip(meanss, weightss, Xs)):
        out[i] = densite_melange(X, weights, means, covs)
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


def mean_IS(Y, gllim, F, r, Nsample=10000):
    Xs = gllim.predict_sample(Y, nb_per_Y=Nsample)
    ws = p_tilde(Xs, Y, F, r) / q(Xs, Y, gllim)
    mask = ~ numpy.isfinite(ws)
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / Nsample:.5f}")
    ws[mask] = 0
    return numpy.sum(Xs * ws[:, :, None], axis=1) / numpy.sum(ws, axis=1, keepdims=True)


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


def main():
    exp, gllim = Experience.setup(context.InjectiveFunction(4), 100, partiel=(0, 1, 2, 3), with_plot=True,
                                  regenere_data=False, with_noise=50, N=100000, method="sobol",
                                  mode="l", init_local=100,
                                  sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)
    X = exp.Xtest[0:100]
    Y = exp.Ytest[0:100]
    r = exp.with_noise
    F = exp.context.F

    Xmean = gllim.predict_high_low(Y)
    Xis = mean_IS(Y, gllim, F, r, Nsample=50000)

    su = exp.mesures.sumup_errors
    nrmse, _, _, _, nrmseY = exp.mesures._nrmse_oneXperY(Xmean, X, Y, F)
    print("Me : ", su(nrmse), "Ye", su(nrmseY))
    nrmse, _, _, _, nrmseY = exp.mesures._nrmse_oneXperY(Xis, X, Y, F)
    print("Me : ", su(nrmse), "Ye", su(nrmseY))


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # _test()
    main()
