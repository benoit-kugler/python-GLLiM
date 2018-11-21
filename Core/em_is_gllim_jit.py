"""Implements a crossed EM - GLLiM algorith to evaluate noise in model.
Diagonal covariance is assumed"""
import logging
import time

import coloredlogs
import numba as nb
import numpy as np

from Core import cython
from Core.gllim import jGLLiM
from Core.probas_helper import densite_melange_precomputed, cholesky_list, _chol_loggausspdf_precomputed, \
    _loggausspdf_diag
from tools import context
import hapke.cython

# GLLiM parameters
Ntrain = 40000
K = 40
init_X_precision_factor = 10
maxIterGlliM = 100
stoppingRatioGLLiM = 0.005

N_sample_IS = 100000

INIT_COV_NOISE = 0.005  # initial noise
INIT_MEAN_NOISE = 0  # initial noise offset
maxIter = 100

NO_IS = False
"""If it's True, dont use Importance sampling"""

HAPKE_MODE = True
"""Use fast implementation for F calculus, bypassing context.F"""


# ------------------------ Linear case ------------------------ #
@nb.njit(cache=True)
def _helper_mu_lin(y, F, K, inverse_current_cov, current_mean):
    return y - F.dot(K).dot(F.T).dot(inverse_current_cov).dot(y - current_mean)


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_lin(Yobs, F, prior_cov, current_mean, current_cov):
    Ny, D = Yobs.shape

    esp_mu = np.zeros((Ny, D))

    invsig = np.linalg.inv(current_cov)
    K = np.linalg.inv(np.linalg.inv(prior_cov) + F.T.dot(invsig).dot(F))

    for i in nb.prange(Ny):
        y = Yobs[i]
        esp_mu[i] = _helper_mu_lin(y, F, K, invsig, current_mean)
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, K, esp_mu


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_diag_lin(Yobs, F, prior_cov, current_mean, current_cov):
    Ny, D = Yobs.shape

    esp_mu = np.zeros((Ny, D))

    invsig = np.diag(1 / current_cov)
    K = np.linalg.inv(np.linalg.inv(prior_cov) + F.T.dot(invsig).dot(F))

    for i in nb.prange(Ny):
        y = Yobs[i]
        esp_mu[i] = _helper_mu_lin(y, F, K, invsig, current_mean)
    maximal_mu = np.sum(esp_mu, axis=0) / Ny
    return maximal_mu, K, esp_mu


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _sigma_step_full_lin(F, K, esp_mu, max_mu):
    base_cov = F.dot(K).dot(F.T)
    Ny, D = esp_mu.shape
    esp_sigma = np.zeros((Ny, D, D))

    for i in nb.prange(Ny):
        u = max_mu - esp_mu[i]
        esp_sigma[i] = u.reshape((-1, 1)).dot(u.reshape((1, -1)))

    maximal_sigma = base_cov + (np.sum(esp_sigma, axis=0) / Ny)
    return maximal_sigma


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _sigma_step_diag_lin(F, K, esp_mu, max_mu):
    base_cov = F.dot(K).dot(F.T)
    Ny, D = esp_mu.shape
    esp_sigma = np.zeros((Ny, D))

    for i in nb.prange(Ny):
        u = max_mu - esp_mu[i]
        esp_sigma[i] = np.square(u)

    maximal_sigma = np.diag(base_cov) + (np.sum(esp_sigma, axis=0) / Ny)
    return maximal_sigma


def _em_step_lin(F, Yobs, prior_cov, current_cov, current_mean):
    ti = time.time()

    _mu_step = _mu_step_diag_lin if current_cov.ndim == 1 else _mu_step_lin

    maximal_mu, K, esp_mu = _mu_step(Yobs, F, prior_cov, current_mean, current_cov)
    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    ti = time.time()
    _sigma_step = _sigma_step_diag_lin if current_cov.ndim == 1 else _sigma_step_full_lin
    maximal_sigma = _sigma_step(F, K, esp_mu, maximal_mu)
    logging.debug(f"Noise covariance estimation done in {time.time()-ti:.3f} s")
    return maximal_mu, maximal_sigma


# -------------------------- General case -------------------------- #
def _init(cont: context.abstractHapkeModel):
    gllim = jGLLiM(K, sigma_type="full", verbose=False)
    Xtrain, Ytrain = cont.get_data_training(Ntrain)
    Ytrain = cont.add_noise_data(Ytrain, covariance=INIT_COV_NOISE, mean=INIT_MEAN_NOISE)  # 0 offset

    m = cont.get_X_uniform(K)
    rho = np.ones(gllim.K) / gllim.K
    precisions = init_X_precision_factor * np.array([np.eye(Xtrain.shape[1])] * gllim.K)
    rnk = gllim._T_GMM_init(Xtrain, 'random',
                            weights_init=rho, means_init=m, precisions_init=precisions)
    gllim.fit(Xtrain, Ytrain, {"rnk": rnk}, maxIter=1)
    return gllim.theta


def _gllim_step(cont: context.abstractHapkeModel, current_noise_cov, current_noise_mean, current_theta):
    ti = time.time()
    gllim = jGLLiM(K, sigma_type="full", stopping_ratio=stoppingRatioGLLiM)
    Xtrain, Ytrain = cont.get_data_training(Ntrain)
    Ytrain = cont.add_noise_data(Ytrain, covariance=current_noise_cov, mean=current_noise_mean)

    gllim.fit(Xtrain, Ytrain, current_theta, maxIter=maxIterGlliM)
    gllim.inversion()
    logging.debug(f"GLLiM step done in {time.time() -ti:.3f} s")
    return gllim


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


# ------------------- WITHOUT IS ------------------- #


def _em_step_NoIS(gllim, compute_F, Yobs, current_cov, *args):
    Xs = gllim.predict_sample(Yobs, nb_per_Y=N_sample_IS)
    mask = ~ np.array([(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in Xs])
    # mask = ~ np.array([[True] * x.shape[0] for x in Xs])
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / N_sample_IS:.5f}")
    ti = time.time()

    FXs = compute_F(Xs, mask)
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
@nb.njit(nogil=True, fastmath=True, cache=True)
def _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y):
    """Returns weights and mu expectancy"""
    q = densite_melange_precomputed(X, weights, means, gllim_chol_covs)
    p_tilde = np.exp(log_p_tilde)
    wsi = p_tilde / q  # Calcul des poids
    G1 = y.reshape((1, -1)) - FX  # estimateur de mu
    esp_mui = _clean_mean_vector(G1, wsi, mask_x)
    return esp_mui, wsi


@nb.njit(nogil=True, parallel=True, fastmath=True)
def _mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov):
    Ny, Ns, D = FXs.shape

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


def _em_step_IS(gllim, compute_Fs, Yobs, current_cov, current_mean):
    Xs = gllim.predict_sample(Yobs, nb_per_Y=N_sample_IS)
    assert np.isfinite(Xs).all()
    mask = ~ np.array([(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in Xs])
    # mask = ~ np.array([[True] * x.shape[0] for x in Xs])
    logging.debug(f"Average ratio of F-non-compatible samplings : {mask.sum(axis=1).mean() / N_sample_IS:.5f}")
    ti = time.time()

    meanss, weightss, _ = gllim._helper_forward_conditionnal_density(Yobs)
    gllim_covs = gllim.SigmakListS

    FXs = compute_Fs(Xs, mask)
    logging.debug(f"Computation of F done in {time.time()-ti:.3f} s")
    ti = time.time()

    assert np.isfinite(FXs).all()

    if current_cov.ndim == 1:
        maximal_mu, ws = _mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov)
    else:
        maximal_mu, ws = _mu_step_full(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov)
    logging.debug(f"Noise mean estimation done in {time.time()-ti:.3f} s")

    ti = time.time()
    _sigma_step = _sigma_step_diag if current_cov.ndim == 1 else _sigma_step_full
    maximal_sigma = _sigma_step(Yobs, FXs, ws, mask, maximal_mu)
    logging.debug(f"Noise covariance estimation done in {time.time()-ti:.3f} s")
    return maximal_mu, maximal_sigma


class NoiseEM:
    """Base class for noise estimation based on EM-like procedure"""

    cont: context.abstractHapkeModel

    def __init__(self, Yobs, cont, cov_type):
        self.Yobs = Yobs
        self.cont = cont
        self.cov_type = cov_type
        global _compute_F
        if HAPKE_MODE:
            _compute_F = self._fast_compute_F

    def compute_Fs(self, Xs, mask):
        D = self.cont.D
        N, Ns, _ = Xs.shape
        FXs = np.empty((N, Ns, D))
        F = self.cont.F
        for i, (X, mask_x) in enumerate(zip(Xs, mask)):
            FX = F(X)
            FX[mask_x == 1, :] = 0  # anyway, ws will be 0
            FXs[i] = FX
        return FXs

    def fast_compute_Hapke(self, Xs, mask):
        N, Ns, _ = Xs.shape

        FXs = hapke.cython.compute_many_Hapke(np.asarray(self.cont.geometries, dtype=np.double), Xs, mask,
                                              self.cont.partiel, self.cont.DEFAULT_VALUES,
                                              self.cont.variables_lims[:, 0],
                                              self.cont.variables_range, self.cont.HAPKE_VECT_PERMUTATION)

        return FXs

    def _get_starting_logging(self):
        return f"""
        Covariance constraint : {self.cov_type}
        Nobs = {len(self.Yobs)} 
        Initial covariance noise : {INIT_COV_NOISE} 
        Initial mean noise : {INIT_MEAN_NOISE}
        """

    def _get_F(self):
        "Should return F information"
        raise NotImplementedError

    def _init_gllim(self):
        "Should return a gllim theta if needed"
        return

    def _gllim_step(self, current_noise_cov, current_noise_mean, current_theta):
        "Should return gllim"
        return

    def _get_em_step(self):
        "Choose which function to use for em_step"
        raise NotImplementedError

    def run(self):
        log = "Starting noise estimation with EM" + self._get_starting_logging()
        logging.info(log)
        Yobs = np.asarray(self.Yobs, dtype=float)
        F = self._get_F()
        em_step = self._get_em_step()

        current_theta = self._init_gllim()
        base_cov = np.eye(self.cont.D) if self.cov_type == "full" else np.ones(self.cont.D)
        current_noise_cov, current_noise_mean = INIT_COV_NOISE * base_cov, INIT_MEAN_NOISE * np.ones(self.cont.D)
        history = [(current_noise_mean.tolist(), current_noise_cov.tolist())]

        for current_iter in range(maxIter):
            gllim = self._gllim_step(current_noise_cov, current_noise_mean, current_theta)

            max_mu, max_sigma = em_step(gllim, F, Yobs, current_noise_cov, current_noise_mean)

            log_sigma = max_sigma if self.cov_type == "diag" else np.diag(max_sigma)
            logging.info(f"""
        Iteration {current_iter+1}/{maxIter}. 
            New estimated OFFSET : {max_mu}
            New estimated COVARIANCE : {log_sigma}""")
            current_noise_cov, current_noise_mean = max_sigma, max_mu
            history.append((current_noise_mean.tolist(), current_noise_cov.tolist()))
        return history


class NoiseEMLinear(NoiseEM):

    def _get_starting_logging(self):
        s = super()._get_starting_logging()
        return " (Linear case)" + s

    def _get_F(self):
        return self.cont.F_matrix

    def _get_em_step(self):
        def em_step(gllim, F, Yobs, current_noise_cov, current_noise_mean):
            return _em_step_lin(F, Yobs, self.cont.PRIOR_COV, current_noise_cov, current_noise_mean)

        return em_step


class NoiseEMGLLiM(NoiseEM):

    def _get_starting_logging(self):
        s = super()._get_starting_logging()
        return " with GLLiM" + s

    def _get_F(self):
        if isinstance(self.cont, context.abstractHapkeModel):
            return self.fast_compute_Hapke
        else:
            return self.compute_Fs

    def _init_gllim(self):
        return _init(self.cont)

    def _gllim_step(self, current_noise_cov, current_noise_mean, current_theta):
        return _gllim_step(self.cont, current_noise_cov, current_noise_mean, current_theta)

    def _get_em_step(self):
        return _em_step_NoIS


class NoiseEMISGLLiM(NoiseEMGLLiM):

    def _get_starting_logging(self):
        s = super()._get_starting_logging()
        return s + f"with IS \n\t\tNSampleIS = {N_sample_IS}"

    def _get_em_step(self):
        return _em_step_IS


def fit(Yobs, cont: context.abstractFunctionModel, cov_type="diag", assume_linear=False):
    Yobs = np.copy(Yobs, "C")  # to ensure Y is contiguous
    if assume_linear:
        fitter = NoiseEMLinear(Yobs, cont, cov_type)
    elif NO_IS:
        fitter = NoiseEMGLLiM(Yobs, cont, cov_type)
    else:
        fitter = NoiseEMISGLLiM(Yobs, cont, cov_type)
    return fitter.run()


# ------------------ maintenance purpose ------------------ #
def _profile():
    global maxIter, Nobs, N_sample_IS, INIT_COV_NOISE, NO_IS
    NO_IS = True
    maxIter = 1
    Nobs = 200
    N_sample_IS = 80000
    cont = context.LabContextOlivine(partiel=(0, 1, 2, 3))

    _, Yobs = cont.get_data_training(Nobs)
    Yobs = cont.add_noise_data(Yobs, covariance=0.005, mean=0.1)
    Yobs = np.copy(Yobs, "C")  # to ensure Y is contiguous

    # fit(Yobs, cont, cov_type="diag")

    def _compute_F2(F, D, Xs2, mask):
        N, Ns, _ = Xs.shape

        return hapke.cython.compute_many_Hapke(np.asarray(cont.geometries, dtype=np.double), Xs2, mask, cont.partiel,
                                               cont.DEFAULT_VALUES, cont.variables_lims[:, 0],
                                               cont.variables_range, cont.HAPKE_VECT_PERMUTATION)

    Xs = np.array([cont.get_X_sampling(N_sample_IS) for i in range(Nobs)])
    mask = np.asarray(np.random.random_sample((Nobs, N_sample_IS)) > 0.8, dtype=int)

    ti = time.time()
    Ys1 = _compute_F(cont.F, cont.D, Xs, mask)
    print("Generique", time.time() - ti)

    ti = time.time()
    Ys2 = _compute_F2(cont.F, cont.D, Xs, mask)
    print("Cython", time.time() - ti)

    assert np.allclose(Ys1, Ys2), "cython compute many error"


def _debug():
    global maxIter, Nobs, N_sample_IS, INIT_COV_NOISE, NO_IS
    NO_IS = False
    maxIter = 2
    Nobs = 20
    N_sample_IS = 1000
    # cont = context.LabContextOlivine(partiel=(0, 1, 2, 3))
    cont = context.LinearFunction()
    INIT_COV_NOISE = 0.005
    _, Yobs = cont.get_data_training(Nobs)
    Yobs = cont.add_noise_data(Yobs, covariance=0.05, mean=2)
    Yobs = np.copy(Yobs, "C")  # to ensure Y is contiguous

    fit(Yobs, cont, cov_type="diag", assume_linear=False)


def _compare():
    D = 10
    L = 4
    Ns = 10000
    Ny = 200

    # mask_x = np.asarray(np.random.random_sample(Ns) > 0.2, dtype=int)
    # mask_x = np.asarray(np.zeros(N),dtype=int)

    T = np.tril(np.ones((L, L))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD
    covs = np.array([cov] * K)

    weightss = np.random.random_sample((Ny, K))
    weightss /= weightss.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((Ny, K, L)) * 12.2

    Yobs = np.random.random_sample((Ny, D))
    FXs = np.random.random_sample((Ny, Ns, D)) * 9
    Xs = np.random.random_sample((Ny, Ns, L)) * 10

    maximal_mu = np.random.random_sample(D)
    mask = np.asarray(np.random.random_sample((Ny, Ns)) > 0.4, dtype=int)

    current_mean = np.random.random_sample(D)
    current_cov = np.arange(D) + 1.2
    # T = np.tril(np.ones((D, D))) * 0.456
    # current_cov = np.dot(T, T.T)

    # X = Xs[0]
    # weights = weightss[0]
    # means = meanss[0]
    # gllim_chol_covs = np.linalg.cholesky(covs)
    # log_p_tilde = np.random.random_sample(Ns)
    # FX = FXs[0]
    # mask_x = mask[0]
    # y = Yobs[0]

    ws = np.random.random_sample((Ny, Ns))

    _mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, covs, current_mean, current_cov)
    # _sigma_step_full(Yobs, FXs, ws, mask, maximal_mu)
    print("jit compiled")

    ti = time.time()
    # S1 = _sigma_step_full_NoIS(Yobs, FXs, mask, maximal_mu)
    S1, V1 = _mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, covs, current_mean, current_cov)
    # S1,V1 = _helper_mu(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
    # S1 = _sigma_step_full(Yobs, FXs, ws, mask, maximal_mu)
    print("jit", time.time() - ti)

    ti = time.time()
    # S2 = cython.sigma_step_full_NoIS(Yobs, FXs, mask, maximal_mu)
    S2, V2 = cython.mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask, covs, current_mean, current_cov)
    # S2,V2 = cython.test(X, weights, means, gllim_chol_covs, log_p_tilde, FX, mask_x, y)
    # S2 = cython.sigma_step_full_IS(Yobs, FXs, ws, mask, maximal_mu)

    print("cython", time.time() - ti)
    # print(V1 - V2)
    print(S1 - S2)
    print(S2)
    assert np.allclose(S1, S2), "sigma step full noIs not same !"
    assert np.allclose(V1, V2), "sigma step full noIs not same !"


def _verifie_cython():
    D = 10
    N = 200000
    T = np.tril(np.ones((D, D))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD
    X = np.random.random_sample((D, N))
    mu = np.random.random_sample(D)
    # U = np.diag(np.arange(4)) +  np.eye(4)  + np.diag([1,2],k=2)

    K = 40

    size = 100000
    covs = np.array([U.T] * K)
    wks = np.random.random_sample((N, K))
    wks /= wks.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((N, K, D))

    S1 = _loggausspdf_diag(X, meanss[0, 0], np.arange(D) + 1.2)
    ti = time.time()
    S1 = _loggausspdf_diag(X, meanss[0, 0], np.arange(D) + 1.2)
    print("numpy ", time.time() - ti)
    ti = time.time()
    S2 = cython.test(X.T, meanss[0, 0], np.arange(D) + 1.2)
    print("cython ", time.time() - ti)
    assert np.allclose(S1, S2.T), " Erreur solve triang"


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # _profile()
    # _debug()
    _compare()
    # _verifie_cython()