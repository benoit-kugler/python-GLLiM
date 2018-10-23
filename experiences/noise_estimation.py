import json
import logging

from matplotlib import pyplot

from tools import context
import numpy as np
import Core.em_is_gllim
import Core.noise_GD

Nobs = 200

BASE_PATH = "/scratch/WORK/IS_EM/history"


def get_path(cont: context.abstractFunctionModel, obs_mode, cov_type, extension):
    tag = _get_observations_tag(obs_mode)
    suff = f"{cont.__class__.__name__}-{tag}-covEstim:{cov_type}-withIS:{not NO_IS}-initCov:{INIT_COV_NOISE}.{extension}"
    return BASE_PATH + suff


def _get_observations_tag(obs_mode):
    if obs_mode == "obs":
        return "trueObs"
    else:
        mean_factor = obs_mode.get("mean", None)
        cov_factor = obs_mode["cov"]
        return f"mean:{mean_factor:.3f}-cov:{cov_factor:.3f}"


def noise_estimator(cont: context.abstractFunctionModel, obs_mode, cov_type, save=False):
    if obs_mode == "obs":
        Yobs = cont.get_observations()
    else:
        mean_factor = obs_mode.get("mean", None)
        cov_factor = obs_mode["cov"]
        _, Yobs = cont.get_data_training(Nobs)
        Yobs = cont.add_noise_data(Yobs, covariance=cov_factor, mean=mean_factor)
    Yobs = np.copy(Yobs, "C")  # to ensure Y is contiguous
    history = fit(Yobs, cont, cov_type=cov_type)
    if not save:
        logging.info("No data saved.")
        return
    path = get_path(cont, obs_mode, cov_type, "json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logging.info(f"History saved in {path}")


def show_history(cont, obs_mode, cov_type):
    path = get_path(cont, obs_mode, cov_type, "json")
    with open(path) as f:
        d = json.load(f)
    mean_history = np.array([h[0] for h in d])
    covs_history = np.array([h[1] for h in d])
    fig = pyplot.figure(figsize=(20, 15))
    axe = fig.add_subplot(121)
    N, D = mean_history.shape
    for i in range(D):
        axe.plot(range(N), mean_history[:, i], label=f"Mean - $G_{ {i+1} }$")
    axe.set_title("Moyenne du bruit")
    axe.set_xlabel("EM-iterations")
    # axe.set_ylim(-0.005, 0.015)
    axe.legend()
    axe = fig.add_subplot(122)
    for i in range(D):
        if cov_type == "full":
            p = covs_history[:, i, i]
        else:
            p = covs_history[:, i]
        axe.plot(range(N), p, label=f"Cov - $G_{ {i+1} }$")
    cov_title = "Covariance (contrainte diagonale)" if cov_type == "diag" else "Covariance (sans contrainte)"
    axe.set_title(cov_title)
    # axe.set_ylim(0,0.01)
    axe.set_xlabel("EM-iterations")
    axe.legend()
    title = f"Initialisation : $\mu = {INIT_MEAN_NOISE}$, $\Sigma = {INIT_COV_NOISE}I_{{D}}$"
    title += f"\n Observations : {_get_observations_tag(obs_mode)}"
    fig.suptitle(title)
    image_path = get_path(cont, obs_mode, cov_type, "png")
    pyplot.savefig(image_path)
    logging.info(f"History plot saved in {image_path}")


def get_last_params(cont, obs_mode, cov_type):
    """Load and returns last values for mean and covariance for the given context"""
    path = get_path(cont, obs_mode, cov_type, "json")
    with open(path) as f:
        d = json.load(f)
    mean, cov = d[-1]
    return np.array(mean), np.array(cov)
