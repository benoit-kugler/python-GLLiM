"""Implements a crossed EM - GLLiM algorith to evaluate noise in model.
Diagonal covariance is assumed"""
import json
import logging
import time

import coloredlogs
import numpy as np
from matplotlib import pyplot

from Core.gllim import jGLLiM
from Core.log_gauss_densities import chol_loggauspdf_diag, chol_loggausspdf, densite_melange
from tools import context

# GLLiM parameters
Ntrain = 50000
K = 40
init_X_precision_factor = 10
maxIterGlliM = 100


N_sample_IS = 100000

INIT_COV_NOISE = 0.005  # initial noise
maxIter = 150


def _G1(Xs, Y, F):
    """Estimateur de mu
    X shape : Ny,N,L
    Y shape : Ny,D
    return shape (Ny,N,D)
    """
    for X, y in zip(Xs, Y):
        yield F(X) - y[None, :]


def _G2(Xs, Y, F, mu):
    """Estimateur de Sigma (diagonal)
    X shape : Ny,N,L
    Y shape : Ny,D
    mu: shape D
    return shape (Ny,N,D)
    """
    for X, y in zip(Xs, Y):
        yield np.square(F(X) + mu[None, :] - y[None, :])


def _G3(Xs, Y, F, mu):
    for X, y in zip(Xs, Y):
        U = F(X) + mu[None, :] - y[None, :]
        yield [u[:, None].dot(u[None, :]) for u in U]


def _gllim_step(cont: context.abstractHapkeModel, current_noise_cov, current_noise_mean, current_theta):
    gllim = jGLLiM(K, sigma_type="full")
    Xtrain, Ytrain = cont.get_data_training(Ntrain)
    Ytrain = cont.add_noise_data(Ytrain, covariance=current_noise_cov, mean=current_noise_mean)

    gllim.fit(Xtrain, Ytrain, current_theta, maxIter=maxIterGlliM)
    gllim.inversion()
    return gllim


def _clean_mean(Gx, w, mask_x):
    mask1 = np.isfinite(Gx).prod(axis=1)
    if Gx.ndim == 3:
        mask1 = mask1.prod(axis=1)
    mask1 = ~ np.asarray(mask1, dtype=bool)
    mask2 = ~ np.isfinite(w)
    mask = mask_x | mask1 | mask2
    w[mask] = 0
    if Gx.ndim == 3:
        Gx[mask, :, :] = 0
        w = w[:, None, None]
    else:
        Gx[mask, :] = 0
        w = w[:, None]
    return np.sum(Gx * w, axis=0) / np.sum(w)


def _em_step(gllim, F, Yobs, current_cov, current_mean):
    Ny, D = Yobs.shape
    Xs = gllim.predict_sample(Yobs, nb_per_Y=N_sample_IS)
    mask = ~ np.array([(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in Xs])
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / N_sample_IS:.5f}")
    ti = time.time()

    meanss, weightss, _ = gllim._helper_forward_conditionnal_density(Yobs)
    gllim_covs = gllim.SigmakListS

    chol_cov = np.linalg.cholesky(current_cov) if current_cov.ndim == 2 else None

    ws, FXs, esp_mu = np.zeros((Ny, N_sample_IS)), np.zeros((Ny, N_sample_IS, D)), np.zeros((Ny, D))
    shape_esp_sigma = (Ny, *current_cov.shape)
    esp_sigma = np.zeros(shape_esp_sigma)
    for i, (y, X, means, weights) in enumerate(zip(Yobs, Xs, meanss, weightss)):
        FX = F(X)  # Besoin plus tard
        mask_x = mask[i]
        FX[mask_x, :] = 0  # de toute façon, ws will be 0
        FXs[i] = FX
        if current_cov.ndim == 1:
            p_tilde = chol_loggauspdf_diag(FX.T + current_mean.T[:, None], y[:, None], current_cov)
        else:
            p_tilde = chol_loggausspdf(FX.T + current_mean.T[:, None], y[:, None], _, cholesky=chol_cov)

        p_tilde = np.exp(p_tilde)  # we used log pdf so far
        q = densite_melange(X, weights, means, gllim_covs)
        ws[i] = p_tilde / q  # Calcul des poids

        G1 = FX - y[None, :]  # estimateur de mu
        esp_mu[i] = _clean_mean(G1, ws[i], mask_x)

    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    G3_func = np.square if current_cov.ndim == 1 else (lambda U: np.array([u[:, None].dot(u[None, :]) for u in U]))

    for i, (y, X, means, weights) in enumerate(zip(Yobs, Xs, meanss, weightss)):
        FX = FXs[i]
        U = FX + maximal_mu[None, :] - y[None, :]
        G3 = G3_func(U)
        esp_sigma[i] = _clean_mean(G3, ws[i], mask[i])

    maximal_sigma = np.sum(esp_sigma, axis=0) / Ny
    return maximal_mu, maximal_sigma


def _init(cont: context.abstractHapkeModel, init_noise_cov):
    gllim = jGLLiM(K, sigma_type="full")
    Xtrain, Ytrain = cont.get_data_training(Ntrain)
    Ytrain = cont.add_noise_data(Ytrain, covariance=init_noise_cov)  # 0 offset

    m = cont.get_X_uniform(K)
    rho = np.ones(gllim.K) / gllim.K
    precisions = init_X_precision_factor * np.array([np.eye(Xtrain.shape[1])] * gllim.K)
    rnk = gllim._T_GMM_init(Xtrain, 'random',
                            weights_init=rho, means_init=m, precisions_init=precisions)
    gllim.fit(Xtrain, Ytrain, {"rnk": rnk}, maxIter=1)
    return gllim.theta


def run_em_is_gllim(Yobs, cont: context.abstractHapkeModel, cov_type="diag"):
    logging.info(f"Starting EM-iS for noise (inital covariance noise : {INIT_COV_NOISE})")

    F = lambda X: cont.F(X, check=False)
    current_theta = _init(cont, INIT_COV_NOISE)
    base_cov = np.eye(cont.D) if cov_type == "full" else np.ones(cont.D)
    current_noise_cov, current_noise_mean = INIT_COV_NOISE * base_cov, np.zeros(cont.D)
    history = []
    for current_iter in range(maxIter):
        gllim = _gllim_step(cont, current_noise_cov, current_noise_mean, current_theta)
        max_mu, max_sigma = _em_step(gllim, F, Yobs, current_noise_cov, current_noise_mean)
        logging.info(f"""Iteration {current_iter+1}/{maxIter}. 
        New estimated OFFSET : {max_mu}
        New estimated COVARIANCE : {max_sigma}""")
        current_noise_cov, current_noise_mean = max_sigma, max_mu
        history.append((current_noise_mean.tolist(), current_noise_cov.tolist()))
    return history


PATHS = {"full": "/scratch/WORK/history_IS-EM_full.json",
         "diag": "/scratch/WORK/history_IS-EM_diag.json"}


def main(cov_type):
    cont = context.LabContextOlivine(partiel=(0, 1, 2, 3))
    Yobs = cont.get_observations()
    history = run_em_is_gllim(Yobs, cont, cov_type=cov_type)
    with open(PATHS[cov_type], "w") as f:
        json.dump(history, f, indent=2)


def show_history(cov_type):
    with open(PATHS[cov_type]) as f:
        d = json.load(f)
    mean_history = np.array([h[0] for h in d])
    covs_history = np.array([h[1] for h in d])
    axe = pyplot.subplot(1, 2, 1)
    D = mean_history.shape[1]
    for i in range(D):
        axe.plot(range(maxIter), mean_history[:, i], label=f"$G_{ {i+1} }$")
    axe.set_title("Moyenne du bruit")
    axe.legend()
    axe = pyplot.subplot(1, 2, 2)
    for i in range(D):
        if cov_type == "full":
            p = covs_history[:, i, i]
        else:
            p = covs_history[:, i]
        axe.plot(range(maxIter), p, label=f"$G_{ {i+1} }$")
    axe.set_title("Covariance (diagonale)")
    axe.legend()
    pyplot.show()


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main("full")
    main("diag")
    # show_history("full")
    # cont = context.LabContextOlivine(partiel=(0, 1, 2, 3))
    # h = cont.geometries
    # np.savetxt("geometries_olivine.txt",h[:,0,:].T,fmt="%.1f")
