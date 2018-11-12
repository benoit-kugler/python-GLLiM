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


def test_em():
    em_is_gllim.maxIter = 2000
    NoiseEstimation.Nobs = 2000
    obs_mode = {"mean": 1, "cov": 0.0001}
    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "is_gllim", assume_linear=True)
    exp.run_noise_estimator(True)
    exp.show_history()


if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # test_gd()
    test_em()
