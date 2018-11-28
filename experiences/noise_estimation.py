import json
import logging
import os.path
import zlib

from matplotlib import pyplot
import coloredlogs

from tools import context
import numpy as np
from Core import em_is_gllim
from Core import noise_GD


class NoiseEstimation:
    Nobs = 200

    BASE_PATH = "/scratch/WORK/NOISE_ESTIMATION/"

    context: context

    def __init__(self, context_class, obs_mode, cov_type, method, assume_linear=False):
        self.context = context_class()
        self.obs_mode = obs_mode
        self.cov_type = cov_type
        self.method = method
        self.assume_linear = assume_linear

    def _is_gllim_tag(self):
        if type(em_is_gllim.INIT_COV_NOISE) is int or type(em_is_gllim.INIT_COV_NOISE) is float:
            initcov = em_is_gllim.INIT_COV_NOISE
        else:
            initcov = zlib.adler32(str(em_is_gllim.INIT_COV_NOISE).encode('utf8'))

        if type(em_is_gllim.INIT_MEAN_NOISE) is int or type(em_is_gllim.INIT_MEAN_NOISE) is float:
            initmean = em_is_gllim.INIT_MEAN_NOISE
        else:
            initmean = zlib.adler32(str(em_is_gllim.INIT_MEAN_NOISE).encode('utf8'))

        if self.assume_linear:
            mode = "LIN"
        else:
            if em_is_gllim.NO_IS:
                mode = "GLLiM"
            else:
                mode = "GLLiM-IS"
        s = f"isGLLiM-mode:{mode}-" \
            f"initCov:{initcov}-initMean:{initmean}"
        return s

    def _gd_tag(self):
        s = f"gradientD-initMean:{noise_GD.INIT_MEAN_NOISE}-Ntrain:{noise_GD.Ntrain:.1E}"
        return s

    def get_path(self, extension):
        obs_tag = self._get_observations_tag()
        method_tag = self._is_gllim_tag() if self.method == "is_gllim" else self._gd_tag()
        if self.assume_linear:
            method_tag += "-LIN-cov:" + str(zlib.adler32(context.LinearFunction.PRIOR_COV))
        suff = f"{self.context.__class__.__name__}-{obs_tag}-covEstim:{self.cov_type}-{method_tag}.{extension}"
        return os.path.join(self.BASE_PATH, suff)

    def _get_observations_tag(self):
        if self.obs_mode == "obs":
            nobs = len(self.context.get_observations())
            return f"trueObs({nobs})"
        else:
            mean_factor = self.obs_mode.get("mean", None)
            cov_factor = self.obs_mode["cov"]
            return f"simuObs({self.Nobs})-mu:{mean_factor:.3f}-Sigma:{cov_factor:.3f}"

    def run_noise_estimator(self, save=False, Yobs=None):
        """If obs_mode is a dict, and Yobs is given, uses it instead of generating new obs
        (usefull to compare).
        Returns Yobs actually used.
        """
        logging.info(f"Starting noise estimation for {self.context.__class__.__name__}")
        if self.obs_mode == "obs":
            Yobs = self.context.get_observations()
        else:
            mean_factor = self.obs_mode.get("mean", None)
            cov_factor = self.obs_mode["cov"]
            if Yobs is None:
                _, Yobs = self.context.get_data_training(self.Nobs)
                Yobs = self.context.add_noise_data(Yobs, covariance=cov_factor, mean=mean_factor)
        Yobs = np.asarray(Yobs, dtype=float, order="C")  # to ensure Y is contiguous
        fit = em_is_gllim.fit if self.method == "is_gllim" else noise_GD.fit
        history = fit(Yobs, self.context, cov_type=self.cov_type, assume_linear=self.assume_linear)
        if not save:
            logging.info("No data saved.")
            return history
        path = self.get_path("json")
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        logging.info(f"History saved in {path}")
        return Yobs

    def _title(self):
        s = "Initialisation : "
        if self.assume_linear:
            s = f"Cas Linéaire. Prior Cov : {context.LinearFunction.PRIOR_COV[0,0]} \n" + s

        if self.method == "is_gllim":
            if em_is_gllim.NO_IS:
                s = "EM-GLLiM \n " + s
            else:
                s = "IS-EM-GLLiM \n " + s
            s += f"$\mu = {em_is_gllim.INIT_MEAN_NOISE}$, $\Sigma = {em_is_gllim.INIT_COV_NOISE}I_{{D}}$"
        else:
            s = "Gradient descent \n " + s
            s += f"$\mu = {noise_GD.INIT_MEAN_NOISE}$"
            if not self.assume_linear:
                s += f"\n N = {noise_GD.Ntrain:,}"
        return s

    def show_history(self, fst_ylims=None, snd_ylims=None):
        path = self.get_path("json")
        with open(path) as f:
            d = json.load(f)
        mean_history = np.array([h[0] for h in d])
        fig = pyplot.figure(figsize=(20, 15))
        axe = fig.add_subplot(121)
        N, D = mean_history.shape
        for i in range(D):
            axe.plot(range(N), mean_history[:, i], label=f"Mean - $G_{ {i+1} }$")
        axe.set_title("Moyenne du bruit")
        axe.set_xlabel("Iterations")
        if fst_ylims is not None:
            axe.set_ylim(fst_ylims)
        axe.legend()
        axe = fig.add_subplot(122)

        if self.method == "gd":
            Jvalues = np.array([h[2] for h in d])
            axe.plot(range(N), Jvalues, label="Distance aux observations")
            axe_title = "Objectif ($J$)"
        else:
            covs_history = np.array([h[1] for h in d])

            for i in range(D):
                if self.cov_type == "full":
                    p = covs_history[:, i, i]
                else:
                    p = covs_history[:, i]
                axe.plot(range(N), p, label=f"Cov - $G_{ {i+1} }$")
            axe_title = "Covariance (contrainte diagonale)" if self.cov_type == "diag" else "Covariance (sans contrainte)"
        axe.set_title(axe_title)
        if snd_ylims is not None:
            axe.set_ylim(snd_ylims)
        axe.set_xlabel("Iterations")
        axe.legend()

        title = self._title()
        title += f"\n Observations : {self._get_observations_tag()}"
        fig.suptitle(title)
        image_path = self.get_path("png")
        pyplot.savefig(image_path)
        logging.info(f"History plot saved in {image_path}")

    def get_last_params(self, average_over=None):
        """Load and returns last values for mean and covariance for the given context
        If average is given, average over given iterations.
        """
        path = self.get_path("json")
        with open(path) as f:
            d = json.load(f)
        logging.debug(f" Noise estimation loaded from {path}")
        average_over = average_over or 1
        means, covs, *_ = zip(*d[:-average_over])
        mean = np.mean(means, axis=0)
        cov = np.mean(covs, axis=0)
        return mean, cov


def launch_tests():
    noise_GD.Ntrain = 100000
    exp = NoiseEstimation(context.MergedLabObservations, "obs", "diag", "gd")

    noise_mean, noise_cov = exp.get_last_params(average_over=500)

    em_is_gllim.Ntrain = 50000
    em_is_gllim.N_sample_IS = 100000
    em_is_gllim.maxIterGlliM = 100
    em_is_gllim.stoppingRatioGLLiM = 0.001
    em_is_gllim.maxIter = 150
    em_is_gllim.INIT_MEAN_NOISE = noise_mean
    em_is_gllim.INIT_COV_NOISE = noise_cov

    # NoiseEstimation.Nobs = 200
    # obs_mode = {"mean": 1, "cov": 0.001}

    # exp = NoiseEstimation(context.LabContextOlivine, obs_mode, "diag", "gd")
    # exp.run_noise_estimator(True)
    # exp.show_history()
    #
    # NoiseEstimation.Nobs = 500
    # exp.run_noise_estimator(True)
    # exp.show_history()

    exp = NoiseEstimation(context.LabContextOlivine, "obs", "diag", "gd")
    exp.run_noise_estimator(save=False)
    # exp.show_history(fst_ylims=(-0.14, 0.14),snd_ylims=(0,0.001))
    # # # #
    # exp = NoiseEstimation(context.LabContextNontronite, "obs", "diag", "gd")
    # # exp.run_noise_estimator(True)
    # exp.show_history(fst_ylims=(-0.14, 0.14),snd_ylims=(0,0.001))
    #
    # exp = NoiseEstimation(context.MergedLabObservations, "obs", "diag", "gd")
    # exp.run_noise_estimator(save=True)
    # exp.show_history()

    # mean_ylims = (-0.12, 0.12)
    # exp = NoiseEstimation(context.LabContextOlivine, "obs", "full", "is_gllim")
    # # exp.run_noise_estimator(True)
    # exp.show_history(fst_ylims=mean_ylims)
    #
    # exp = NoiseEstimation(context.LabContextNontronite, "obs", "full", "is_gllim")
    # # exp.run_noise_estimator(True)
    # exp.show_history(fst_ylims=mean_ylims)
    #
    # exp = NoiseEstimation(context.MergedLabObservations, "obs", "full", "is_gllim")
    # # exp.run_noise_estimator(True)
    # exp.show_history(fst_ylims=mean_ylims)



if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")

    # launch_tests()
    # em_is_gllim.Ntrain = 20000
    # em_is_gllim.N_sample_IS = 50000
    # em_is_gllim.maxIterGlliM = 30
    # em_is_gllim.stoppingRatioGLLiM = 0.001
    # em_is_gllim.maxIter = 1
    #
    # NoiseEstimation.Nobs = 1000
    # obs_mode = {"mean":1,"cov":0.001}
    # exp = NoiseEstimation(context.InjectiveFunction(2),obs_mode,"diag","gd")
    # exp.run_noise_estimator(True)
    # exp.show_history()
    # raise
    # _, Yobs = exp.context.get_data_training(exp.Nobs)
    # Yobs = exp.context.add_noise_data(Yobs, covariance=obs_mode["cov"], mean=obs_mode["mean"])
    #
    # b, _ = exp.get_last_params()
    # _,Ytrain = exp.context.get_data_training(100000)
    # print((noise_GD.verifie_J(b, Yobs, Ytrain)))
    # print((noise_GD.verifie_J(obs_mode["mean"] * np.ones(exp.context.D), Yobs, Ytrain)))
