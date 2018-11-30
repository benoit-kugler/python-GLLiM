"""Implements a crossed EM - GLLiM algorith to evaluate noise in model.
Diagonal covariance is assumed"""
import logging
import time

import coloredlogs
import numba as nb
import numpy as np

from Core import em_is_gllim
from Core.probas_helper import densite_melange_precomputed, cholesky_list, _chol_loggausspdf_precomputed, \
    _loggausspdf_diag
from tools import context


@nb.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _clean_mean_vector(Gx, w, mask_x):
    N, L = Gx.shape

    mask1 = np.empty(N, dtype=np.bool_)
    for i in range(N):
        mask1[i] = not np.isfinite(Gx[i]).all()

    mask2 = ~ np.isfinite(w)
    mask = (mask_x | mask1 | mask2) != 0

    w2 = np.copy(w).reshape((-1, 1))
    w2[mask, :] = 0
    Gx[mask] = np.zeros(L)

    s = np.sum(w2)
    if s == 0:
        return np.zeros(L)

    return np.sum(Gx * w2, axis=0) / s


@nb.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _clean_mean_matrix(Gx, w, mask_x):
    N, L, _ = Gx.shape

    mask1 = np.empty(N, dtype=np.bool_)
    for i in range(N):
        mask1[i] = not np.isfinite(Gx[i]).all()

    mask2 = ~ np.isfinite(w)
    mask = (mask_x | mask1 | mask2) != 0

    w[mask] = 0
    Gx[mask] = np.zeros((L, L))

    s = np.sum(w)
    if s == 0:
        return np.zeros((L, L))

    w2 = w.reshape((-1, 1, 1))
    return np.sum(Gx * w2, axis=0) / s


@nb.njit(nogil=True, cache=True)
def extend_array(vector, Ns):
    D = vector.shape[0]
    extended = np.zeros((Ns, D))
    for i in range(Ns):
        extended[i] = np.copy(vector)
    return extended



# --------------------------------- WITH IS --------------------------------- #
@nb.njit(nogil=True, fastmath=True, cache=True)
def _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y):
    """Returns weights and mu expectancy"""
    q = densite_melange_precomputed(X, weights, means, gllim_chol_covs)
    av_log = np.mean(log_p_tilde)  # re sclaing to avoid numerical issues
    p_tilde = np.exp(log_p_tilde - av_log)

    wsi = p_tilde / q  # Calcul des poids

    # effective_sample_size = np.sum(wsi) ** 2 / np.sum(np.square(wsi))
    # print("Effective sample size : ", effective_sample_size)

    G1 = y.reshape((1, -1)) - FX  # estimateur de mu
    esp_mui = _clean_mean_vector(G1, wsi, mask_x)
    return esp_mui, wsi


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov):
    Ny, Ns, D = FXs.shape
    L = Xs.shape[1]
    gllim_chol_covs = cholesky_list(gllim_covs)
    ws = np.zeros((Ny, Ns))
    esp_mu = np.zeros((Ny, D))

    current_mean_broad = extend_array(current_mean, Ns)

    for i in range(Ny):
        y = Yobs[i]
        X = Xs[i]
        means = meanss[i]
        weights = weightss[i]
        FX = FXs[i]
        mask_x = mask[i]
        arg = FX + current_mean_broad
        log_p_tilde = _loggausspdf_diag(arg.T, y, current_cov)
        a, b = _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
        esp_mu[i], ws[i] = a, b
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, ws




@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_diag_with_prior(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov, prior_cov):
    Ny, Ns, D = FXs.shape
    L = Xs.shape[2]
    gllim_chol_covs = cholesky_list(gllim_covs)
    ws = np.zeros((Ny, Ns))
    esp_mu = np.zeros((Ny, D))

    current_mean_broad = extend_array(current_mean, Ns)

    chol_prior_cov = np.linalg.cholesky(prior_cov)

    for i in range(Ny):
        y = Yobs[i]
        X = Xs[i]
        means = meanss[i]
        weights = weightss[i]
        FX = FXs[i]
        mask_x = mask[i]
        arg = FX + current_mean_broad
        log_p_tilde = _loggausspdf_diag(arg.T, y, current_cov)
        log_p_tilde += _chol_loggausspdf_precomputed(X.T, np.zeros(L), chol_prior_cov)
        a, b = _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
        esp_mu[i], ws[i] = a, b
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, ws


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_full(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov):
    Ny, Ns, D = FXs.shape

    gllim_chol_covs = cholesky_list(gllim_covs)
    chol_cov = np.linalg.cholesky(current_cov)
    ws = np.zeros((Ny, Ns))
    esp_mu = np.zeros((Ny, D))

    current_mean_broad = extend_array(current_mean, Ns)

    for i in nb.prange(Ny):
        y = Yobs[i]
        X = Xs[i]
        means = meanss[i]
        weights = weightss[i]
        FX = FXs[i]
        mask_x = mask[i]
        arg = FX + current_mean_broad
        log_p_tilde = _chol_loggausspdf_precomputed(arg.T, y, chol_cov)
        esp_mu[i], ws[i] = _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, ws


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_full_with_prior(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov, prior_cov):
    Ny, Ns, D = FXs.shape
    L = Xs.shape[2]

    gllim_chol_covs = cholesky_list(gllim_covs)
    chol_cov = np.linalg.cholesky(current_cov)
    ws = np.zeros((Ny, Ns))
    esp_mu = np.zeros((Ny, D))

    current_mean_broad = extend_array(current_mean, Ns)

    chol_prior_cov = np.linalg.cholesky(prior_cov)

    for i in nb.prange(Ny):
        y = Yobs[i]
        X = Xs[i]
        means = meanss[i]
        weights = weightss[i]
        FX = FXs[i]
        mask_x = mask[i]
        arg = FX + current_mean_broad
        log_p_tilde = _chol_loggausspdf_precomputed(arg.T, y, chol_cov)
        log_p_tilde += _chol_loggausspdf_precomputed(X.T, np.zeros(L), chol_prior_cov)
        esp_mu[i], ws[i] = _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, ws


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _sigma_step_diag(Yobs, FXs, ws, mask, maximal_mu):
    Ny, Ns, D = FXs.shape
    maximal_mu_broadcast = extend_array(maximal_mu, Ns)
    esp_sigma = np.zeros((Ny, D))

    for i in range(Ny):
        y = Yobs[i]
        FX = FXs[i]
        U = FX + maximal_mu_broadcast - extend_array(y, Ns)
        G3 = np.square(U)
        esp_sigma[i] = _clean_mean_vector(G3, ws[i], mask[i])

    maximal_sigma = np.sum(esp_sigma, axis=0) / Ny
    return maximal_sigma


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _sigma_step_full(Yobs, FXs, ws, mask, maximal_mu):
    Ny, Ns, D = FXs.shape
    maximal_mu_broadcast = extend_array(maximal_mu, Ns)
    esp_sigma = np.zeros((Ny, D, D))

    for i in range(Ny):
        y = Yobs[i]
        FX = FXs[i]
        U = FX + maximal_mu_broadcast - extend_array(y, Ns)
        G3 = np.zeros((Ns, D, D))
        for j in range(Ns):
            u = U[j]
            G3[j] = u.reshape((-1, 1)).dot(u.reshape((1, -1)))

        esp_sigma[i] = _clean_mean_matrix(G3, ws[i], mask[i])

    maximal_sigma = np.sum(esp_sigma, axis=0) / Ny
    return maximal_sigma


def _em_step_IS(gllim, compute_Fs, get_X_mask, Yobs, current_cov, current_mean, prior_cov=None):
    Xs = gllim.predict_sample(Yobs, nb_per_Y=em_is_gllim.N_sample_IS)
    assert np.isfinite(Xs).all()
    mask = get_X_mask(Xs)
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / em_is_gllim.N_sample_IS:.5f}")
    ti = time.time()

    meanss, weightss, _ = gllim._helper_forward_conditionnal_density(Yobs)
    gllim_covs = gllim.SigmakListS

    FXs = compute_Fs(Xs, mask)
    logging.debug(f"Computation of F done in {time.time()-ti:.3f} s")
    ti = time.time()

    assert np.isfinite(FXs).all()

    args = Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov
    if current_cov.ndim == 1 and prior_cov is None:
        maximal_mu, ws = _mu_step_diag(*args)
    elif current_cov.ndim == 1:
        maximal_mu, ws = _mu_step_diag_with_prior(*args,prior_cov)
    elif current_cov.ndim == 2 and prior_cov is None:
        maximal_mu, ws = _mu_step_full(*args)
    else:
        maximal_mu, ws = _mu_step_full_with_prior(*args, prior_cov)

    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    ti = time.time()
    _sigma_step = _sigma_step_diag if current_cov.ndim == 1 else _sigma_step_full
    maximal_sigma = _sigma_step(Yobs, FXs, ws, mask, maximal_mu)
    logging.debug(f"Noise covariance estimation done in {time.time()-ti:.3f} s")
    return maximal_mu, maximal_sigma


# ------------------- WITHOUT IS ------------------- #

def _em_step_NoIS_cython(gllim, compute_Fs, get_X_mask, Yobs, current_cov, *args):
    Xs = gllim.predict_sample(Yobs, nb_per_Y=em_is_gllim.N_sample_IS)
    mask = get_X_mask(Xs)
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / em_is_gllim.N_sample_IS:.5f}")
    ti = time.time()

    FXs = compute_Fs(Xs, mask)
    logging.debug(f"Computation of F done in {time.time()-ti:.3f} s")
    ti = time.time()

    maximal_mu = cython.mu_step_NoIS(Yobs, FXs, mask)
    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    ti = time.time()
    _sigma_step = cython.sigma_step_diag_NoIS if current_cov.ndim == 1 else cython.sigma_step_full_NoIS
    maximal_sigma = _sigma_step(Yobs, FXs, mask, maximal_mu)
    logging.debug(f"Noise covariance estimation done in {time.time()-ti:.3f} s")
    return maximal_mu, maximal_sigma


# --------------------------------- WITH IS --------------------------------- #

def _em_step_IS_cython(gllim, compute_Fs, get_X_mask, Yobs, current_cov, current_mean):
    Xs = gllim.predict_sample(Yobs, nb_per_Y=em_is_gllim.N_sample_IS)
    mask = get_X_mask(Xs)
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / em_is_gllim.N_sample_IS:.5f}")
    ti = time.time()

    meanss, weightss, _ = gllim._helper_forward_conditionnal_density(Yobs)
    gllim_covs = gllim.SigmakListS

    FXs = compute_Fs(Xs, mask)
    logging.debug(f"Computation of F done in {time.time()-ti:.3f} s")
    ti = time.time()

    assert np.isfinite(FXs).all()

    if current_cov.ndim == 1:
        maximal_mu, ws = cython.mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                                current_cov, parallel = em_is_gllim.PARALLEL)
    else:
        maximal_mu, ws = cython.mu_step_full_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                                current_cov)
    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    ti = time.time()
    _sigma_step = cython.sigma_step_diag_IS if current_cov.ndim == 1 else cython.sigma_step_full_IS
    maximal_sigma = _sigma_step(Yobs, FXs, ws, mask, maximal_mu)
    logging.debug(f"Noise covariance estimation done in {time.time()-ti:.3f} s")
    return maximal_mu, maximal_sigma



class NoiseEMISGLLiM(em_is_gllim.NoiseEMGLLiM):

    def _get_starting_logging(self):
        s = super()._get_starting_logging()
        return s + f"with IS \n\tNSampleIS = {em_is_gllim.N_sample_IS}"

    def _get_em_step(self):
        if hasattr(self.cont, "PRIOR_COV"):
            logging.info("Using Gaussian prior on X")
            return lambda *args: _em_step_IS(*args, prior_cov=self.cont.PRIOR_COV)
        return _em_step_IS


def fit(Yobs, cont: context.abstractFunctionModel, cov_type="diag", assume_linear=False):
    Yobs = np.copy(Yobs, "C")  # to ensure Y is contiguous
    if assume_linear or em_is_gllim.NO_IS:
        raise ValueError("Linear case or No IS case already supported in em_is_gllim. Nothing new here !")
    else:
        fitter = NoiseEMISGLLiM(Yobs, cont, cov_type)
    return fitter.run()
