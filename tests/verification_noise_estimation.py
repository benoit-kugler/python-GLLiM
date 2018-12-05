import logging

import coloredlogs

from Core import noise_GD
from Core import em_is_gllim
from old import em_is_gllim_jit

from experiences.noise_estimation import NoiseEstimation
from experiences import noise_estimation
from tools import context


def test_gd_linear():
    old_name = NoiseEstimation.BASE_PATH

    NoiseEstimation.BASE_PATH += "LinearyGaussian/"
    noise_GD.INIT_MEAN_NOISE = 0
    noise_GD.maxIter = 500
    NoiseEstimation.Nobs = 2000
    obs_mode = {"mean": 1, "cov": 0.0001}

    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "gd", assume_linear=True)
    Yobs = exp.run_noise_estimator(True)
    exp.show_history()

    exp.assume_linear = False
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()

    NoiseEstimation.BASE_PATH = old_name


def test_em_linear():
    """Compare EM IS GLLiM, EM GLLiM, and linear formulas"""
    old_name = NoiseEstimation.BASE_PATH
    NoiseEstimation.BASE_PATH += "LinearyGaussian/"

    em_is_gllim.maxIter = 30
    NoiseEstimation.Nobs = 1000
    obs_mode = {"mean": 1, "cov": 0.1}

    exp = NoiseEstimation(context.LinearFunction, obs_mode, "diag", "is_gllim", assume_linear=True)
    # Yobs = exp.run_noise_estimator(True)
    exp.show_history()

    exp.assume_linear = False
    # em_is_gllim.NO_IS = True
    # exp.run_noise_estimator(save=True, Yobs=Yobs)
    # exp.show_history()


    # em_is_gllim.NO_IS = False
    # em_is_gllim.N_sample_IS = 1000
    # exp.run_noise_estimator(save=True, Yobs=Yobs)
    # exp.show_history()
    #
    #
    # em_is_gllim.N_sample_IS = 10000
    # exp.run_noise_estimator(save=True, Yobs=Yobs)
    # exp.show_history()
    #
    # em_is_gllim.N_sample_IS = 50000
    # exp.run_noise_estimator(save=True, Yobs=Yobs)
    # exp.show_history()
    #
    # em_is_gllim.N_sample_IS = 100000
    # exp.run_noise_estimator(save=True, Yobs=Yobs)
    # exp.show_history()

    NoiseEstimation.BASE_PATH = old_name

def test_em_easy():
    em_is_gllim.maxIter = 50
    em_is_gllim.maxIterGlliM = 10

    NoiseEstimation.Nobs = 400
    obs_mode = {"mean": 0.2, "cov": 0.05}

    exp = NoiseEstimation(context.EasyFunction, obs_mode, "diag", "is_gllim", assume_linear=False)
    Yobs = exp.run_noise_estimator(True)
    exp.show_history()




    em_is_gllim.N_sample_IS = 10000
    exp.run_noise_estimator(True, Yobs=Yobs)
    exp.show_history()

    em_is_gllim.N_sample_IS = 50000
    exp.run_noise_estimator(True, Yobs=Yobs)
    exp.show_history()


    em_is_gllim.NO_IS = True
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()

    em_is_gllim.N_sample_IS = 100000
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()

    exp.Nobs = 2000
    exp.run_noise_estimator(save=True)
    exp.show_history()


def test_em():
    em_is_gllim.maxIter = 50
    em_is_gllim.maxIterGlliM = 10

    noise_GD.maxIter = 200

    NoiseEstimation.Nobs = 200
    obs_mode = {"mean": 0.2, "cov": 0.05}


    exp = NoiseEstimation(context.InjectiveFunction(3), obs_mode, "diag", "gd", assume_linear=False)
    Yobs = exp.run_noise_estimator(True)
    exp.show_history()


    em_is_gllim.NO_IS = True
    exp = NoiseEstimation(context.InjectiveFunction(3), obs_mode, "diag", "is_gllim", assume_linear=False)
    exp.run_noise_estimator(True, Yobs=Yobs)
    exp.show_history()

    em_is_gllim.NO_IS = False
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()

    em_is_gllim.N_sample_IS = 50000
    exp.run_noise_estimator(save=True, Yobs=Yobs)
    exp.show_history()



def main():
    # test_gd_linear()
    # test_em_linear()
    try:
        test_em_easy()
    finally:
        test_em()

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
