import logging

import coloredlogs

from Core import noise_GD, em_is_gllim
from experiences.noise_estimation import NoiseEstimation
from tools import context


def test_gd():
    noise_GD.INIT_MEAN_NOISE = -5
    NoiseEstimation.Nobs = 2000
    obs_mode = {"mean": 1, "cov": 0.0001}
    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "gd", assume_linear=True)
    # exp.run_noise_estimator(True)
    exp.show_history()


def test_em_linear():
    """Compare EM IS GLLiM, EM GLLiM, and linear formulas"""
    NoiseEstimation.BASE_PATH += "LinearyGaussian/"
    em_is_gllim.maxIter = 300
    NoiseEstimation.Nobs = 200
    obs_mode = {"mean": 1, "cov": 0.1}

    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "is_gllim", assume_linear=True)
    Yobs = exp.run_noise_estimator(True)
    exp.show_history()

    exp.assume_linear = False
    em_is_gllim.NO_IS = True
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()

    em_is_gllim.NO_IS = False
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()


def test_em():
    em_is_gllim_jit.maxIter = 300
    noise_GD.maxIter = 200
    NoiseEstimation.Nobs = 2000
    em_is_gllim_jit.NO_IS = True
    obs_mode = {"mean": 1, "cov": 0.1}
    # exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "is_gllim", assume_linear=False)
    # exp.run_noise_estimator(True)
    # exp.show_history()

    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "gd", assume_linear=False)
    exp.run_noise_estimator(True)
    exp.show_history()


if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # test_gd()
    test_em_linear()
    # test_em()
