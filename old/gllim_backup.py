"""
Gllim model in python

__author__ = R.Juge & S.Lathuiliere & B. Kugler

The equation numbers refer to _High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables_A. Deleforge 2015

"""
import logging
import time
import warnings

import coloredlogs
import numpy as np
import scipy
from numpy.linalg import inv
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

from Core import probas_helper, mixture_merging
from Core.probas_helper import chol_loggausspdf, densite_melange, dominant_components, chol_loggausspdf_iso, \
    GMM_sampling, chol_loggausspdf_diag
from tools import regularization
import Core.cython

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CovarianceTypeError(NotImplementedError):

    def __init__(self, gamma_type=None, sigma_type=None):
        super().__init__(f"This covariance type Gamma ! {gamma_type}; Sigma : {sigma_type} is not supported !")


class WrongContextError(ValueError):
    pass


def _inv_sym_def(S):
    """Computes S inverse with cholesky decomposition for numerical stability"""
    D = S.shape[0]
    S_chol = np.linalg.cholesky(S)
    i_chol = scipy.linalg.solve_triangular(S_chol,
                                           np.eye(D), lower=True)
    Si = np.dot(i_chol.T, i_chol)
    return Si


def get_full_covariances(covariances_, covariance_type, K=None, N_features=None):
    if covariance_type == 'spherical':
        return covariances_.reshape(K, 1, 1) * np.repeat(np.eye(N_features).reshape(1, N_features, N_features), K,
                                                         axis=0)
    elif covariance_type == 'tied':
        return np.repeat(covariances_.reshape(1, N_features, N_features), K, axis=0)
    elif covariance_type == 'diag':
        return np.array([np.diag(sk) for sk in covariances_])
    else:
        return covariances_


class MyGMM(GaussianMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1,
                 random_state=None, warm_start=False, weights_init=None, precisions_init=None, means_init=None,
                 verbose=0, verbose_interval=10, init_params='random', track=False):
        super().__init__(n_components=n_components, tol=tol, reg_covar=reg_covar,
                         max_iter=max_iter, n_init=n_init, init_params=init_params,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval, covariance_type=covariance_type,
                         weights_init=weights_init,
                         means_init=means_init,
                         precisions_init=precisions_init)
        self.log_likelihoods = []
        self.current_iter_ll = []
        self.current_iter = 0
        self.track = track
        self.track_params = []

    @property
    def last_ll(self):
        assert self.n_init == 1
        return self.log_likelihoods[0][-1]

    def _m_step(self, Y, log_resp):
        super()._m_step(Y, log_resp)
        self.current_iter_ll.append(self.log_likelihood(Y))

    def log_likelihood(self, Y):
        vec_log_prob, _ = self._estimate_log_prob_resp(Y)
        return vec_log_prob.sum()

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        self.current_iter = n_iter
        if self.track:
            self.track_params.append((self.weights_, self.means_, self.full_covariances_))
        if self.verbose >= 0:
            logging.debug(f"Iteration {n_iter} : Log-likelihood = {self.current_iter_ll[-1]:.3f}")

    def _print_verbose_msg_init_end(self, ll):
        self.log_likelihoods.append(self.current_iter_ll)
        self.current_iter_ll = []

    def _print_verbose_msg_init_beg(self, n_init):
        pass

    @property
    def full_covariances_(self):
        K, N_features = self.means_.shape
        return get_full_covariances(self.covariances_, self.covariance_type, K, N_features)


DEFAULT_REG_COVAR = 1e-08
DEFAULT_STOPPING_RATIO = 0.001


class OldGLLiM():
    ''' Gaussian Locally-Linear Mapping'''

    def __init__(self, K_in, Lw=0, sigma_type='iso', gamma_type='full',
                 verbose=True,
                 reg_covar=DEFAULT_REG_COVAR, stopping_ratio=DEFAULT_STOPPING_RATIO):

        self.K = K_in
        self.Lw = Lw
        self.sigma_type = sigma_type
        self.gamma_type = gamma_type
        self.reg_covar = reg_covar
        self.stopping_ratio = stopping_ratio
        self.verbose = verbose
        self.track_theta = False
        self.nb_init_GMM = 1  # Number of init made by GMM when fit is init with it

    def start_track(self):
        self.track_theta = True
        self.track = []

    def _init_from_dict(self, dic):
        if "A" in dic:
            self.AkList = np.array(dic['A'])
        if "b" in dic:
            self.bkList = np.array(dic['b'])
        if 'c' in dic:
            ckList = np.array(dic['c'])
            self.ckList_T = ckList[:, :self.Lt]
            if self.Lw == 0:
                self.ckList_W = np.zeros((self.K, 0))
            else:
                self.ckList_W = ckList[:, -self.Lw:]
        if "Gamma" in dic:
            GammakList = np.array(dic['Gamma'])
            if self.gamma_type == "iso":
                self.GammakList_T = GammakList[:, 0, 0]
                logging.debug("Gamma_T init from first coeff of given matrix")
            elif self.gamma_type == 'full':
                self.GammakList_T = GammakList[:, :self.Lt, :self.Lt]
            else:
                raise CovarianceTypeError
            self.GammakList_W = GammakList[:, -self.Lw:, -self.Lw:]
        if "pi" in dic:
            self.pikList = np.array(dic["pi"])
        if "Sigma" in dic:
            self.SigmakList = np.array(dic["Sigma"])
        if self.verbose:
            used_keys = set(dic.keys()) & {"A", "b", "c", "Gamma", "pi", "Sigma"}
            logging.debug(f"Init from parameters {used_keys}")

    @property
    def theta(self):
        return dict(
            pi=self.pikList.tolist(),
            c=self.ckList.tolist(),
            Gamma=self.GammakList.tolist(),
            A=self.AkList.tolist(),
            b=self.bkList.tolist(),
            Sigma=self.SigmakList.tolist()
        )

    @property
    def dict_julia(self):
        assert self.Lw == 0
        dic = self.theta
        dic["K"], dic["L"], dic["D"] = self.K, self.L, self.D
        dic["Sigma"] = self.full_SigmakList
        return dic

    @property
    def current_ll(self):
        return self.LLs_[-1]

    @property
    def loglikelihoods(self):
        """Returns LL over the iterations"""
        return self.LLs_

    @property
    def L(self):
        return self.Lt + self.Lw

    @property
    def AkList_W(self):
        if self.Lw == 0:
            return np.zeros((self.K, self.D, 0))
        return self.AkList[:, :, -self.Lw:]

    @property
    def AkList_T(self):
        return self.AkList[:, :, :self.Lt]

    @property
    def ckList(self):
        return np.concatenate((self.ckList_T, self.ckList_W), axis=1)

    @property
    def GammakList(self):
        if self.gamma_type == "iso":
            gammas = [g * np.eye(self.Lt) for g in self.GammakList_T]
        elif self.gamma_type == 'full':
            gammas = self.GammakList_T
        else:
            raise CovarianceTypeError

        if self.Lw == 0:
            return np.array(gammas)

        return np.array([
            np.block(
                [[Gammak_t, np.zeros((self.Lt, self.Lw))], [np.zeros((self.Lw, self.Lt)), Gammak_w]])
            for Gammak_t, Gammak_w in zip(gammas, self.GammakList_W)
        ])

    @property
    def full_SigmakList(self):
        if self.sigma_type == "iso":
            sitype = 'spherical'
        elif self.sigma_type == "diag":
            sitype = "diag"
        elif self.sigma_type == "full":
            sitype = "full"
        else:
            raise CovarianceTypeError
        return get_full_covariances(self.SigmakList, sitype, self.K, self.D)

    def _T_GMM_init(self, T, init_mode, **theta):
        """Performs GMM init_mode, initialized with theta if given. Returns rnk"""
        if self.verbose:
            logging.debug("Initialization of posterior with GaussianMixture")
        start_time_EMinit = time.time()

        gmm = GaussianMixture(n_components=self.K, covariance_type='full', max_iter=5,
                              n_init=self.nb_init_GMM, init_params=init_mode, **theta)
        gmm.fit(T)
        rnk = gmm.predict_proba(T)  # shape N , K
        if self.verbose:
            logging.debug("--- {} seconds for EM initialization---".format(time.time() - start_time_EMinit))
        return rnk

    def _default_init(self):
        # Add S covariances
        self.SkList_W = np.zeros((self.K, self.Lw, self.Lw))

        # Means and Covariances of W fixed for non-identifiability issue
        self.ckList_W = np.zeros((self.K, self.Lw))
        if self.gamma_type == 'full':
            self.GammakList_W = np.array([np.eye(self.Lw)] * self.K)
        elif self.gamma_type == 'iso':
            self.GammakList_W = np.ones(self.K)
        elif self.gamma_type == "diag":
            self.GammakList_W = np.ones((self.K, self.Lw))
        else:
            raise CovarianceTypeError

        self.pikList = np.ones(self.K) / self.K
        self.AkList = np.ones((self.K, self.D, self.L))
        self.bkList = np.zeros((self.K, self.D))
        self.ckList_T = np.ones((self.K, self.Lt)) * np.arange(self.K)[:, None] / self.K
        if self.gamma_type == 'full':
            self.GammakList_T = np.array([np.identity(self.Lt)] * self.K)
        elif self.gamma_type == 'iso':
            self.GammakList_T = np.ones(self.K)
        elif self.gamma_type == "diag":
            self.GammakList_T = np.ones((self.K, self.Lt))
        else:
            raise CovarianceTypeError

        if self.sigma_type == 'full':
            self.SigmakList = np.array([np.identity(self.D)] * self.K)
        elif self.sigma_type == 'iso':
            self.SigmakList = np.ones(self.K)
        elif self.sigma_type == "diag":
            self.SigmakList = np.ones((self.K, self.D))
        else:
            raise CovarianceTypeError

    def init_fit(self, T, Y, init):
        """Initialize model parameters. Three cases are supported :
            - init = 'kmeans' :  GMM initialization, itself with kmeans initialization
            - init = None : initialization with basic values (zeros, identity)
            - init = 'random' : GMM initialization, itself with random initialization
            - init = rnk : Array of clusters probabilities : skip GMM init.
            - init = theta , where theta is a dict of Gllim parameters (with Sigma shape compatible with sigmae_type)
        Remark : At the end, all that matter are rnk, since fit start by maximization.
        """
        init = () if init is None else init
        self.Lt = T.shape[1]
        self.D = Y.shape[1]

        self._default_init()

        if init in ['random', 'kmeans']:
            self.rnk = self._T_GMM_init(T, init)

        elif 'rnk' in init:
            if self.verbose:
                logging.debug('Initialization with given rnk')
            self.rnk = np.array(init['rnk'])
            self._rnk_init = np.array(self.rnk)
            assert self.rnk.shape == (T.shape[0], self.K)
        elif type(init) is dict:
            self._init_from_dict(init)
            _, logrnk = self._compute_rnk(T, Y)
            self.rnk = np.exp(logrnk)
        else:
            _, logrnk = self._compute_rnk(T, Y)
            self.rnk = np.exp(logrnk)

        self.rkList = self.rnk.sum(axis=0)

    def _remove_empty_cluster(self):
        keep = ~ (self.rkList == 0 + np.isinf(self.rkList))
        cpt = np.sum(~ keep)
        if not cpt:
            return
        if self.verbose is not None:
            logging.debug("{} cluster(s) removed".format(cpt))
        self.K -= cpt
        self.rkList = self.rkList[keep]
        self.AkList = self.AkList[keep]
        self.bkList = self.bkList[keep]
        self.ckList_T = self.ckList_T[keep]
        self.ckList_W = self.ckList_W[keep]
        self.pikList = self.pikList[keep]
        self.GammakList_T = self.GammakList_T[keep]
        self.GammakList_W = self.GammakList_W[keep]
        self.SigmakList = self.SigmakList[keep]
        self.rnk = self.rnk[:, keep]

    def _add_numerical_stability(self, matrixlist, cov_type):
        if cov_type == 'iso':
            return matrixlist + self.reg_covar
        elif cov_type == "diag":
            return matrixlist + self.reg_covar
        elif cov_type == 'full':
            dim = matrixlist.shape[1]
            return matrixlist + np.array([np.eye(dim) * self.reg_covar] * self.K)
        else:
            raise CovarianceTypeError

    def _get_SkList_X(self, SkList_W):
        return np.array([
            np.block(
                [[np.zeros((self.Lt, self.Lt)), np.zeros((self.Lt, self.Lw))],
                 [np.zeros((self.Lw, self.Lt)), Sk_w]])
            for Sk_w in SkList_W
        ])

    def _compute_rW_Z(self, Y, T):
        """
        Compute parameters of gaussian distribution W knowing Z : munk_W and Sk_W

        :param Y: shape (N,D)
        :param T: shape (N,Lt)
        :return: munk_W shape (K,N,Lw) Sk_W shape (K,Lw,Lw)
        """

        if self.Lw == 0:
            N = Y.shape[0]
            return np.zeros((self.K, N, 0)), np.zeros((self.K, 0, 0))

        AkList_W = self.AkList_W
        if self.gamma_type == "iso":
            ginv = np.array([(1 / g) * np.eye(self.Lw) for g in self.GammakList_W])
        elif self.gamma_type == "diag":
            ginv = np.array([np.diag(1 / g) for g in self.GammakList_W])
        else:
            ginv = inv(self.GammakList_W)

        ATSinv = np.matmul(AkList_W.transpose((0, 2, 1)), inv(self.full_SigmakList))
        Sk_W = inv(ginv + np.matmul(ATSinv, AkList_W))
        u = np.array([Ak.dot(T.T) for Ak in self.AkList_T])
        d = Y.T - u - self.bkList[:, :, None]

        e = np.matmul(ATSinv, d) + np.matmul(ginv, self.ckList_W[:, :, None])

        munk_W = np.matmul(Sk_W, e).transpose((0, 2, 1))
        return munk_W, Sk_W

    def _compute_GammaT(self, T, ckList_T):
        N = T.shape[0]
        # Evite la répition du calcul de a et b
        for k, ck, rk in zip(range(self.K), ckList_T, self.rkList):
            a = np.sqrt(self.rnk[:, k]).reshape((1, N))
            b = T.T - ck.reshape((self.Lt, 1))
            c = a * b
            mat = np.dot(c, c.T) / rk
            if self.gamma_type == 'iso':
                if self.Lt == 0:
                    mat = 0
                else:
                    trace = mat.trace(axis1=0, axis2=1)
                    mat = trace / self.Lt
            elif self.gamma_type == 'diag':
                mat = np.diag(mat)
            yield mat

    def _compute_Sigma(self, Xnk, Y, AkList, bkList, SkList_W):
        """Eq (38)."""
        if self.sigma_type == "iso":
            SigmaList = np.zeros(self.K)
        elif self.sigma_type == "diag":
            SigmaList = np.zeros((self.K, self.D))
        elif self.sigma_type == "full":
            SigmaList = np.zeros((self.K, self.D, self.D))
        else:
            raise CovarianceTypeError(sigma_type=self.sigma_type)

        for k, Ak, bk, rk in zip(range(self.K), AkList, bkList, self.rkList):
            coefs = self.rnk[:, k] / rk
            diffSigma1 = (Y - (Ak.dot(Xnk[:, k, :].T)).T - bk.reshape((1, self.D))).T
            diffSigma = np.sqrt(coefs).T * diffSigma1

            if self.sigma_type == 'iso':
                stmp = np.sum((diffSigma ** 2), axis=1)
                # isotropic sigma
                SigmaList[k] = np.sum(stmp) / self.D
            elif self.sigma_type == 'full':
                dS_large = diffSigma.T[:, :, None]
                pro = np.matmul(dS_large, dS_large.transpose((0, 2, 1)))
                pro = np.array(pro, dtype='double')
                SigmaList[k] = pro.sum(axis=0)
            elif self.sigma_type == "diag":
                s = np.sum(diffSigma ** 2, axis=1)
                SigmaList[k] = s
            else:
                raise NotImplementedError("Covariance type unknown !")

            assert np.isfinite(SigmaList[k]).all(), "Sigma matrix is not finite !"

        if self.Lw == 0:
            return SigmaList

        # Calcul de ASAw
        Akwlist = AkList[:, :, -self.Lw:]
        AS = np.matmul(Akwlist, SkList_W)
        ASAwk = np.matmul(AS, Akwlist.transpose((0, 2, 1)))

        if self.sigma_type == 'iso':
            trace = ASAwk.trace(axis1=1, axis2=2)
            r = SigmaList + (trace / self.D)
        elif self.sigma_type == 'full':
            r = SigmaList + ASAwk
        elif self.sigma_type == "diag":
            r = SigmaList + np.array([np.diag(m) for m in ASAwk])
        else:
            raise CovarianceTypeError
        return r

    def _compute_rnk(self, T, Y):
        N = T.shape[0]
        logrnk = np.empty((N, self.K))

        gamma_f_log = {"iso": chol_loggausspdf_iso, "full": chol_loggausspdf, "diag": chol_loggausspdf_diag}[
            self.gamma_type]

        for (k, Ak, Ak_W, bk, ck_W, ck_T, pik, gammak_T, gammak_W, sigmak) in zip(range(self.K), self.AkList,
                                                                                  self.AkList_W, self.bkList,
                                                                                  self.ckList_W, self.ckList_T,
                                                                                  self.pikList,
                                                                                  self.GammakList_T, self.GammakList_W,
                                                                                  self.SigmakList):

            c = gamma_f_log(T.T, ck_T, gammak_T)

            cnk = np.array([ck_W] * N)
            X = np.concatenate((T, cnk), axis=1)
            y_mean = np.dot(Ak, X.T) + bk.reshape((self.D, 1))

            if self.gamma_type == "iso":  # full gamma
                gammak_W = gammak_W * np.eye(self.Lw)
            elif self.gamma_type == "diag":
                gammak_W = np.diag(gammak_W)

            if self.sigma_type == "iso":
                if self.Lw == 0:
                    d = chol_loggausspdf_iso(Y.T, y_mean, sigmak)
                else:
                    sigmak = sigmak * np.eye(self.D)  # full sigmak
                    aga = np.dot(np.dot(Ak_W, gammak_W), Ak_W.T)
                    sigmak = sigmak + aga
                    d = chol_loggausspdf(Y.T, y_mean, sigmak)
            elif self.sigma_type == "diag":
                if self.Lw == 0:
                    d = chol_loggausspdf_diag(Y.T, y_mean, sigmak)
                else:
                    sigmak = np.diag(sigmak)  # full sigmak
                    aga = np.dot(np.dot(Ak_W, gammak_W), Ak_W.T)
                    sigmak = sigmak + aga
                    d = chol_loggausspdf(Y.T, y_mean, sigmak)
            elif self.sigma_type == "full":
                if not self.Lw == 0:
                    aga = np.dot(np.dot(Ak_W, gammak_W), Ak_W.T)
                    sigmak = sigmak + aga
                d = chol_loggausspdf(Y.T, y_mean, sigmak)
            else:
                raise CovarianceTypeError(sigma_type=self.sigma_type)
            # Pondération (experimental, not convincing)
            # c = c * self.D
            # d = d * (self.L / self.D)
            rnks = np.log(pik) + c + d
            logrnk[:, k] = rnks
            # print("Log rnk (d c d+c) : ", d[0], c[0], c[0] + d[0])

        lognormrnk = logsumexp(logrnk, axis=1, keepdims=True)
        logrnk -= lognormrnk
        print(lognormrnk)
        assert (logrnk <= 0).all()
        return lognormrnk, logrnk


    def _compute_Ak(self, Xnk, Y, SkList_X):
        xk_bars = (self.rnk[:, :, None] * Xnk).sum(axis=0) / self.rkList[:, None]

        yk_bars = [np.sum(self.rnk[:, k] * Y.T, axis=1) / rk for k, rk in enumerate(self.rkList)]  # (36)

        AkList = np.zeros((self.K, self.D, self.L))
        for k, rk, xk, yk in zip(range(self.K), self.rkList, xk_bars, yk_bars):
            # print(self.rnk[:,k],(X-xk).T)
            X_stark = (np.sqrt(self.rnk[:, k])) * (Xnk[:, k, :] - xk).T  # (33)
            X_stark /= np.sqrt(rk)
            # print(X_stark)
            Y_stark = (np.sqrt(self.rnk[:, k])) * (Y - yk).T  # (34)
            Y_stark /= np.sqrt(rk)

            XXt = np.dot(X_stark, X_stark.T)
            XXt_stark = SkList_X[k] + XXt

            YXt_stark = np.dot(Y_stark, X_stark.T)
            try:
                i = np.linalg.pinv(XXt_stark)
                A = np.dot(YXt_stark, i)
                assert np.isfinite(A).all()
            except (np.linalg.LinAlgError, AssertionError) as e:
                logging.warning(f"Warning ! {e} at iter. {self.current_iter}-> A set to 0")
                AkList[k] = np.zeros((self.D, self.L))
            else:
                AkList[k] = A

        return AkList

    def _compute_bk(self, Y, Xnk, AkList):
        return np.array([np.sum(self.rnk[:, k].T * (Y - (Ak.dot(Xnk[:, k, :].T)).T).T, axis=1) / rk for k, Ak, rk in
                         zip(range(self.K), AkList, self.rkList)])  # (37)

    def _compute_ck(self, T):
        return np.dot(self.rnk.T, T) / self.rkList[:, np.newaxis]  # (29)

    def compute_next_theta(self, T, Y):
        """Compute M steps. Return the result. Usefull to implement SAEM algorithm"""
        N = T.shape[0]

        munk, SkList_W = self._compute_rW_Z(Y, T)

        Xnk = np.concatenate((np.array([T] * self.K), munk), axis=2).transpose((1, 0, 2))  # Shape (N,K,L)

        pikList = self.rkList / N  # (28)

        ckList_T = self._compute_ck(T)

        GammakList_T = np.array(list(self._compute_GammaT(T, ckList_T)))
        GammakList_T = self._add_numerical_stability(GammakList_T, self.gamma_type)  # numerical stability

        # M-mapping-step
        SkList_X = self._get_SkList_X(SkList_W)

        AkList = self._compute_Ak(Xnk, Y, SkList_X)

        bkList = self._compute_bk(Y, Xnk, AkList)

        SigmakList = self._compute_Sigma(Xnk, Y, AkList, bkList, SkList_W)
        SigmakList = self._add_numerical_stability(SigmakList, self.sigma_type)

        return pikList, ckList_T, GammakList_T, AkList, bkList, SigmakList

    def stopping_criteria(self, maxIter):
        """Return true if we should stop"""
        if self.current_iter < 3:
            return False
        if self.current_iter > maxIter:
            return True
        delta_total = max(self.LLs_) - min(self.LLs_)
        delta = self.current_ll - self.LLs_[-2]
        return delta < (self.stopping_ratio * delta_total)

    def fit(self, T, Y, init, maxIter=100):
        '''fit the Gllim
           # Arguments
            X: low dimension targets as a Numpy array
            Y: high dimension features as a Numpy array
            maxIter: maximum number of EM algorithm iterations
            init: None, 'kmeans', 'random' or theta
        '''
        N, L = T.shape
        _, D = Y.shape
        if self.verbose is not None:
            logging.info("{} initialization... (N = {}, L = {} , D = {}, K = {})".format(self.__class__.__name__,
                                                                                         N, L, D, self.K))
        self.init_fit(T, Y, init)

        if self.verbose is not None:
            logging.info("Done. GLLiM fitting...")
        self.current_iter = 0
        self.LLs_ = []
        converged = False

        start_time_EM = time.time()

        while not converged:
            self._remove_empty_cluster()

            if self.verbose:
                logging.debug("M - Step...")

            self.pikList, self.ckList_T, self.GammakList_T, self.AkList, self.bkList, self.SigmakList = \
                self.compute_next_theta(T, Y)

            if self.verbose:
                logging.debug("E - Step...")

            lognormrnk, logrnk = self._compute_rnk(Y, T)

            self.rnk = np.exp(logrnk)
            self.rkList = self.rnk.sum(axis=0)

            # Log likelihood of (X,Y)
            ll = np.sum(lognormrnk)  # EVERY EM Iteration THIS MUST INCREASE
            self.end_iter_callback(ll)
            self.current_iter += 1
            converged = self.stopping_criteria(maxIter)

        if self.verbose:
            logging.debug(f"Final log-likelihood : {self.LLs_[self.current_iter - 1]}")
            logging.debug(f" Converged in {self.current_iter} iterations")

        if self.verbose is not None:
            t = int(time.time() - start_time_EM)
            logging.info("--- {} mins, {} secs for fit ---".format(t // 60, t - 60 * (t // 60)))

    def end_iter_callback(self, loglikelihood):
        if self.verbose is not None:
            logging.debug(f"Iteration {self.current_iter} : Log-likelihood = {loglikelihood:.3f} ")

        self.LLs_.append(loglikelihood)
        if self.track_theta:  # Save parameters history
            self.track.append(self.theta)

    def inversion(self):
        """ Bayesian inversion of the parameters"""
        start_time_inversion = time.time()

        self.ckListS = np.array([Ak.dot(ck) + bk for Ak, bk, ck in zip(self.AkList, self.bkList, self.ckList)])  # (9)

        self.GammakListS = np.array([sig + Ak.dot(gam).dot(Ak.T) for sig, gam, Ak in
                                     zip(self.full_SigmakList, self.GammakList, self.AkList)])  # (10)

        self.SigmakListS = np.empty((self.K, self.L, self.L))
        self.AkListS = np.empty((self.K, self.L, self.D))
        self.bkListS = np.empty((self.K, self.L))

        for k, sig, gam, Ak, ck, bk in zip(range(self.K), self.SigmakList, self.GammakList, self.AkList, self.ckList,
                                           self.bkList):
            if self.sigma_type == 'iso':
                i = 1 / sig * Ak
            elif self.sigma_type == 'full':
                i = _inv_sym_def(sig)
                i = np.dot(i, Ak)
            else:
                raise CovarianceTypeError

            if np.allclose(Ak, np.zeros((self.D, self.L))):
                sigS = gam
                bS = ck
            else:
                ig = _inv_sym_def(gam)
                sigS = _inv_sym_def(ig + (Ak.T).dot(i))  # (14)
                bS = sigS.dot(ig.dot(ck) - i.T.dot(bk))  # (13)

            aS = sigS.dot(i.T)  # (12)

            self.SigmakListS[k] = sigS
            self.AkListS[k] = aS
            self.bkListS[k] = bS

        if self.verbose is not None:
            logging.debug(f"GLLiM inversion done in {time.time()-start_time_inversion:.3f} s")

    def inversion_with_null_sigma(self):
        # Inversion step
        if self.sigma_type == "iso":
            self.SigmakList = np.ones(self.K) * 1e-12
        else:
            self.SigmakList = np.array([np.eye(self.D) * 1e-12] * self.K)
        self.inversion()

    @property
    def norm2_SigmaSGammaInv(self):
        return np.array([np.linalg.norm(x, 2) for x in
                         np.matmul(self.SigmakListS, inv(self.GammakList))])

    def _helper_forward_conditionnal_density(self, Y):
        """
        Compute the mean Ak*Y + Bk and the quantities alpha depending of Y in (7)
        :param Y: shape (N,D)
        :return: mean shape(N,K,L) alpha shape (N,K) , normalisation shape (N,1)
        """
        N = Y.shape[0]
        Y = Y.reshape((N, self.D))
        YT = np.array(Y.T, dtype=float)

        proj = np.empty((self.L, N, self.K))  # AkS * Y + BkS
        logalpha = np.zeros((N, self.K))  # log N(ckS,GammakS)(Y)

        for (k, pik, Ak, bk, ck, Gammak) in zip(range(self.K), self.pikList, self.AkListS,
                                                self.bkListS, self.ckListS, self.GammakListS):
            proj[:, :, k] = Ak.dot(YT) + np.expand_dims(bk, axis=1)
            logalpha[:, k] = np.log(pik) + chol_loggausspdf(YT, ck.reshape((self.D, 1)), Gammak)

        log_density = logsumexp(logalpha, axis=1, keepdims=True)
        logalpha -= log_density
        alpha, normalisation = np.exp(logalpha), np.exp(log_density)
        return proj.transpose((1, 2, 0)), alpha, normalisation

    def predict_high_low(self, Y, with_covariance=False):
        """Forward prediction.
        If with_covariance, returns covariance matrix of the mixture, shape (len(Y),L,L)"""
        proj, alpha, _ = self._helper_forward_conditionnal_density(Y)
        if with_covariance:
            Xpred, Covs = probas_helper.mean_cov_melange(alpha, proj, self.SigmakListS)
            return Xpred, Covs
        else:
            Xpred = probas_helper.mean_melange(alpha, proj)
            return Xpred

    def predict_cluster(self, X, with_covariance=False):
        """Backward prediction
        If with_covariance is True, the importance of one cluster is computed with the height of gaussian as well."""
        N = X.shape[0]
        prob = np.empty((self.K, N))
        if with_covariance:
            chols = np.linalg.cholesky(self.full_SigmakList)
            dets = np.sum(np.log(np.array([np.diag(c) for c in chols])), axis=1)
        for k, ck, Gammak, pik in zip(range(self.K), self.ckList, self.GammakList, self.pikList):
            r = chol_loggausspdf(X.T, ck[:, None], Gammak) + np.log(pik)
            if with_covariance:  # poids = pik / sqrt( det(Sigma))
                r = r - dets[k]
            prob[k] = r
        choice = np.argmax(prob, axis=0)
        prob = np.exp(prob)
        prob = prob / prob.sum(axis=0)
        return choice, prob.T

    def X_density(self, X_points, marginals=None):
        """Return density of X, evaluated at X_points.
        If marginals is given, compute marginal density. In this case, X_points needs to have the marginal dimension.
        """
        if (not marginals) and not X_points.shape[1] == self.L:
            raise WrongContextError("Dimension of X samples doesn't match the choosen Lw")

        if marginals:
            means = self.ckList[:, marginals]  # K , len(marginals)
            covs = self.GammakList[:, marginals, :][:, :, marginals]  # K, len(marginals), len(marginals)
        else:
            means = self.ckList
            covs = self.GammakList

        return densite_melange(X_points, self.pikList, means, covs)

    def forward_density(self, Y, X_points, marginals=None, sub_densities=0):
        """Return conditionnal density of X knowing Y, evaluated at X_points.
        Return shape (N ,len(X_points) ).
        If marginals is given, compute marginal density. In this case, X_points needs to have the marginal dimension.
        Is sub_densities is a non negative integer, returns the density of sub_densitites dominant components."""

        if (not marginals) and not X_points.shape[1] == self.L:
            raise WrongContextError("Dimension of X samples doesn't match the choosen Lw")
        proj, alpha, _ = self._helper_forward_conditionnal_density(Y)

        NX, D = X_points.shape
        N = Y.shape[0]
        if marginals:
            proj = proj[:, :, marginals]  # len(marginals) , N , K
            covs = self.SigmakListS[:, marginals, :][:, :, marginals]  # K, len(marginals), len(marginals)
        else:
            covs = self.SigmakListS

        densites = np.empty((N, NX))
        sub_dens = np.empty((sub_densities, N, NX))
        t = time.time()
        for n, meann, alphan in zip(range(N), proj, alpha):
            densites[n] = densite_melange(X_points, alphan, meann, covs)
            if sub_densities:
                dominants = dominant_components(alphan, meann, covs)[0:sub_densities]
                for i, (_, w, m, c) in enumerate(dominants):
                    sub_dens[i, n] = np.exp(chol_loggausspdf(X_points.T, m.reshape((D, 1)), c)) * w
        if self.verbose:
            logging.debug("Density calcul time {:.3f}".format(time.time() - t))

        return densites, sub_dens

    def modal_prediction(self, Y, components=None, threshold=None, sort_by="height"):
        """Returns components of conditionnal mixture by descending order of importance.
        If given, limits to components values.
        If threshold is given, gets rid of components with weight <= threshold
        Priority on components"""
        proj, alpha, _ = self._helper_forward_conditionnal_density(Y)
        covs = self.SigmakListS
        chols = np.linalg.cholesky(covs)
        det_covs = np.prod(np.array([np.diag(c) for c in chols]), axis=1)
        N = Y.shape[0]
        lc = components or self.K
        threshold = threshold if (components is None) else None
        X, heights, weights = [], [], []
        for n, meann, alphan in zip(range(N), proj, alpha):
            dominants = dominant_components(alphan, meann, covs, threshold=threshold,
                                            sort_by=sort_by, dets=det_covs)[0:lc]
            if len(dominants) == 0:
                max_w = max(alphan)
                logging.error("Warning ! No prediction for this threshold (best weight : {:.2e}!".format(max_w))
                hs, ws, xs = np.empty((0,)), np.empty((0,)), np.empty((0,))
            else:
                hs, ws, xs, _ = zip(*dominants)
            weights.append(np.array(ws))
            heights.append(np.array(hs))
            X.append(np.array(xs))
        return X, heights, weights

    def predict_sample(self, Y, nb_per_Y=10):
        """Compute law of X knowing Y and nb_per_Y points following this law"""
        proj, alpha, _ = self._helper_forward_conditionnal_density(Y)
        ti = time.time()
        covs = self.SigmakListS
        s = GMM_sampling(proj, alpha, covs, nb_per_Y)
        logging.debug(f"Sampling from mixture ({len(Y)} series of {nb_per_Y}) done in {time.time()-ti:.3f} s")
        return s

    @staticmethod
    def monte_carlo_esperance(samples, centres):
        """Compute approximate E[X | X proche de ck] / E[X proche de ck]
        samples shape : (N,L) centres shape (K,L)
        """
        choix = np.square(samples[:, None] - centres[None, :]).sum(axis=2).argmin(axis=1)
        crible = (choix[:, None] == np.arange(len(centres)))
        denom = crible.sum(axis=0)[:, None]
        esp_rapp = (samples[:, None, :] * crible[:, :, None]).sum(axis=0) / denom
        return esp_rapp, choix

    def clustered_prediction(self, Y, F, nb_predsMax=3, size_sampling=10000,
                             agg_method="mean"):
        """Compute prediction of several x per y.
        Number of x to predict is choosen according to F criteria."""
        meanss, weightss, _ = self._helper_forward_conditionnal_density(Y)
        # sampless = self._sample_from_mixture(meanss, weightss, size)  # too large when N is big
        Xmeans = probas_helper.mean_melange(meanss, weightss)  # avoid recomp
        preds = []
        N = len(Y)
        k_choosen = []
        agg = np.max if agg_method == "max" else np.mean
        for n, X, weights, y, xmean, means in zip(range(N), meanss, weightss, Y, Xmeans, meanss):
            samples = GMM_sampling(means[None, :], weights[None, :], self.SigmakListS, size_sampling)[0]
            y_accuracy, xs = [np.square(F(xmean[None, :])[0] - y).sum()], [xmean[None, :]]
            for nb_preds in range(2, nb_predsMax + 1):
                w = regularization.WeightedKMeans(nb_preds)
                try:
                    labels, score, centers = w.fit_predict_score(X, weights, None)
                except ValueError:
                    err, Xpreds = np.inf, None
                else:
                    Xpreds, _ = self.monte_carlo_esperance(samples, centers)
                    if not np.isfinite(Xpreds).all():
                        err, Xpreds = np.inf, None
                    else:
                        ys = F(Xpreds)
                        err = agg(np.square(ys - y).sum(axis=1))
                y_accuracy.append(err)
                xs.append(Xpreds)
            best_K = np.argmin(y_accuracy)
            k_choosen.append(best_K)
            preds.append(xs[best_K])
            if n and n % 100 == 0:
                logging.debug(f"Prediction {n}/{N} done.")
        props = (np.array(k_choosen)[:, None] == np.arange(0, nb_predsMax)[None, :]).sum(axis=0) / N
        logging.info(f"Number of prediction proporations : {props}")
        return preds

    def merged_prediction(self, Y):
        meanss, weightss, _ = self._helper_forward_conditionnal_density(Y)
        ti = time.time()
        Xmean, Covs, Xweight = mixture_merging.merge_predict(weightss, meanss, self.SigmakListS)
        logging.info(f"Merging of GMM mixture done in {time.time() - ti:.3f} s")
        return Xmean, Covs, Xweight


class jGLLiM(OldGLLiM):
    """Estimate parameters with joint Gaussian Mixture equivalence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.sigma_type == 'full' and self.gamma_type == 'full' and self.Lw == 0):
            raise WrongContextError("Joint Gaussian mixture can only be used with Lw = 0, "
                                    "and full covariances matrix")

    @staticmethod
    def GMM_to_GLLiM(rho, m, V, L):
        """
        Compute GLLiM parameters from equivalent GMM model

        :param rho: Weights
        :param m: Means
        :param V: Covariances
        :param L: Dimension of X vectors
        :return: (pi,c,Gamma,A,b,Sigma)
        """
        LplusD = V.shape[1]
        pi = rho
        c = m[:, 0:L]
        Gamma = V[:, 0:L, 0:L]
        g_inv = inv(Gamma)
        V_xy = V[:, 0:L, L:LplusD]
        V_xyT = V_xy.transpose((0, 2, 1))
        A = np.matmul(V_xyT, g_inv)
        K, D, _ = A.shape
        b = m[:, L:LplusD] - np.matmul(A, c[:, :, None]).reshape((K, D))
        Sigma = V[:, L:LplusD, L:LplusD] - np.matmul(np.matmul(A, Gamma), A.transpose((0, 2, 1)))
        return {"pi": pi, "c": c, "Gamma": Gamma, "A": A, "b": b, "Sigma": Sigma}

    @staticmethod
    def GLLiM_to_GGM(pi, c, Gamma, A, b, Sigma):
        """
        Compute GMM parameters of equivalent model

        :param pi: Weights
        :param c: Means of X knowing Z
        :param Gamma: Covariances of X knowing Z
        :param A: Mapping from X to Y
        :param b: idem
        :param Sigma: Covariance of the mapping
        :return: (rho,m,V)
        """
        K = pi.shape[0]
        rho = np.array(pi)
        my = np.matmul(A, c[:, :, None])[:, :, 0] + b
        m = np.concatenate((c, my), axis=1)
        AG = np.matmul(A, Gamma)
        Vy = Sigma + np.matmul(AG, A.transpose((0, 2, 1)))
        V = np.array([
            np.block([[Gamma[k], AG[k].T], [AG[k], Vy[k]]]) for k in range(K)
        ])
        return {"rho": rho, "m": m, "V": V}

    def _Gmm_setup(self, T, Y, maxIter):
        first_theta = self.compute_next_theta(T, Y)  # theta from rnk
        jGMM_params = self.GLLiM_to_GGM(*first_theta)
        precisions_chol = _compute_precision_cholesky(jGMM_params["V"], "full")
        precisions = np.matmul(precisions_chol, precisions_chol.transpose((0, 2, 1)))
        TY = np.concatenate((T, Y), axis=1)

        verbose = {None: -1, False: 0, True: 1}[self.verbose]
        self.Gmm = MyGMM(n_components=self.K, n_init=1, max_iter=maxIter, reg_covar=self.reg_covar,
                         tol=self.stopping_ratio,
                         weights_init=jGMM_params["rho"], means_init=jGMM_params["m"], precisions_init=precisions,
                         verbose=verbose, track=self.track_theta)
        return TY, self.Gmm

    @property
    def current_iter(self):
        if hasattr(self, "Gmm"):
            return self.Gmm.current_iter
        return 0

    def fit(self, T, Y, init, maxIter=100):
        """Use joint GMM model
           # Arguments
            X: low dimension targets as a Numpy array
            Y: high dimension features as a Numpy array
            maxIter: maximum number of EM algorithm iterations
            init: None, 'kmeans', 'random' or theta
        """
        N, L = T.shape
        _, D = Y.shape
        if self.verbose is not None:
            logging.info("{} initialization... (N = {}, L = {} , D = {}, K = {})".format(self.__class__.__name__,
                                                                                         N, L, D, self.K))
        self.init_fit(T, Y, init)
        TY, Gmm = self._Gmm_setup(T, Y, maxIter)

        start_time_EM = time.time()

        Gmm.fit(TY)
        self.LLs_ = Gmm.log_likelihoods[0]

        if self.verbose is not None:
            t = int(time.time() - start_time_EM)
            logging.info("jGMM fit done in {} mins, {} secs".format(t // 60, t - 60 * (t // 60)))

        if self.track_theta:
            self.track = self.track_from_gmm(Gmm)

        rho, m, V = Gmm.weights_, Gmm.means_, Gmm.covariances_
        self._init_from_dict(self.GMM_to_GLLiM(rho, m, V, self.L))

    def track_from_gmm(self, Gmm):
        tolist = lambda rho, m, V: {c: v.tolist() for c, v in
                                    self.GMM_to_GLLiM(rho, m, V, self.L).items()}

        return [tolist(rho, m, V) for (rho, m, V) in Gmm.track_params]


# -------------------- DEBUG -------------------- #
def _compare(g: OldGLLiM, T, Y, Lw):
    N = T.shape[0]
    g.init_fit(T, Y, None)

    AkList_W, AkList_T, GammakList_W, SigmakList, bkList, ckList_W = (g.AkList_W, g.AkList_T, g.GammakList_W,
                                                                      g.full_SigmakList, g.bkList, g.ckList_W)

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")
    ti = time.time()
    a1, b1 = g._compute_rW_Z(Y, T)
    print("python ", time.time() - ti)

    a2, b2 = np.zeros((N, Lw)), np.zeros((Lw, Lw))
    if g.gamma_type == "iso" and g.sigma_type == "full":
        f = Core.cython._compute_rW_Z_GIso_SFull
    elif g.gamma_type == "diag" and g.sigma_type == "full":
        f = Core.cython._compute_rW_Z_GDiag_SFull
    elif g.gamma_type == "full" and g.sigma_type == "full":
        f = Core.cython._compute_rW_Z_GFull_SFull
    elif g.gamma_type == "iso" and g.sigma_type == "diag":
        f = Core.cython._compute_rW_Z_GIso_SFull
    elif g.gamma_type == "diag" and g.sigma_type == "diag":
        f = Core.cython._compute_rW_Z_GDiag_SFull
    elif g.gamma_type == "full" and g.sigma_type == "diag":
        f = Core.cython._compute_rW_Z_GFull_SFull
    elif g.gamma_type == "iso" and g.sigma_type == "iso":
        f = Core.cython._compute_rW_Z_GIso_SFull
    elif g.gamma_type == "diag" and g.sigma_type == "iso":
        f = Core.cython._compute_rW_Z_GDiag_SFull
    elif g.gamma_type == "full" and g.sigma_type == "iso":
        f = Core.cython._compute_rW_Z_GFull_SFull
    else:
        raise CovarianceTypeError(g.gamma_type, g.sigma_type)

    ti = time.time()
    f(Y, T, AkList_W[0], AkList_T[0], GammakList_W[0],
      SigmakList[0], bkList[0], ckList_W[0], a2, b2)
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(a1, a2), "munk"
    assert np.allclose(b1, b2), "Sk_W"


def _compare2(g: OldGLLiM, T, Y, Lw):
    g.init_fit(T, Y, None)

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")

    munk, SkList_W = g._compute_rW_Z(Y, T)
    Xnk = np.concatenate((np.array([T] * g.K), munk), axis=2).transpose((1, 0, 2))  # Shape (N,K,L)
    SkList_X = g._get_SkList_X(SkList_W)

    ti = time.time()
    a1 = g._compute_Ak(Xnk, Y, SkList_X)
    print("python ", time.time() - ti)

    ti = time.time()
    a2 = Core.cython.test_Ak(Y, Xnk, g.rnk, SkList_X)
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(a1, a2), "error in Ak"


def _compare2bis(g: OldGLLiM, T, Y, Lw):
    g.init_fit(T, Y, None)

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")

    ti = time.time()
    a1 = g._compute_ck(T)
    print("python ", time.time() - ti)

    ti = time.time()
    a2 = Core.cython.test_ck(T, g.rnk)
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(a1, a2), "error in ck"


def _compare3(g: OldGLLiM, T, Y, Lw):
    g.init_fit(T, Y, None)

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")

    munk, SkList_W = g._compute_rW_Z(Y, T)
    Xnk = np.concatenate((np.array([T] * g.K), munk), axis=2).transpose((1, 0, 2))  # Shape (N,K,L)
    SkList_X = g._get_SkList_X(SkList_W)
    Ak = g.AkList

    ti = time.time()
    a1 = g._compute_bk(Y, Xnk, Ak)
    a1 = a1[0]
    print("python ", time.time() - ti)

    ti = time.time()
    a2 = Core.cython.test_ak(Y, Xnk[:, 0, :], g.rnk[:, 0], Ak[0])
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(a1, a2), "error in bk"


def _compare4(g: OldGLLiM, T, Y, Lw):
    N, D = Y.shape
    g.init_fit(T, Y, None)

    munk, SkList_W = g._compute_rW_Z(Y, T)
    Xnk = np.concatenate((np.array([T] * g.K), munk), axis=2).transpose((1, 0, 2))  # Shape (N,K,L)
    SkList_X = g._get_SkList_X(SkList_W)
    AkList = g.AkList
    bkList = g.bkList

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")
    ti = time.time()
    a1 = g._compute_Sigma(Xnk, Y, AkList, bkList, SkList_W)
    print("python ", time.time() - ti)

    ti = time.time()
    if g.sigma_type == "full":
        a2 = np.zeros((D, D))
        Core.cython._compute_Sigmak_SFull(Y, Xnk[:, 0, :], g.rnk[:, 0], g.rnk[:, 0].sum(),
                                          AkList[0], bkList[0], SkList_W[0], np.zeros(D),
                                          a2)
    elif g.sigma_type == "diag":
        a2 = np.zeros(D)
        Core.cython._compute_Sigmak_SDiag(Y, Xnk[:, 0, :], g.rnk[:, 0], g.rnk[:, 0].sum(),
                                          AkList[0], bkList[0], SkList_W[0], a2)
    elif g.sigma_type == "iso":
        a2 = Core.cython._compute_Sigmak_SIso(Y, Xnk[:, 0, :], g.rnk[:, 0], g.rnk[:, 0].sum(),
                                              AkList[0], bkList[0], SkList_W[0])
    else:
        raise CovarianceTypeError(g.gamma_type, g.sigma_type)
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(a1, a2), "Sigmak error"
    # assert np.allclose(b1,b2), "Sk_W"


def _compare_complet(g: OldGLLiM, T, Y, Lw):
    g.init_fit(T, Y, None)

    AkList_W, AkList_T, GammakList_W, SigmakList, bkList, ckList_W = (g.AkList_W, g.AkList_T, g.GammakList_W,
                                                                      g.SigmakList, g.bkList, g.ckList_W)
    GammakList_T, ckList_T = g.GammakList_T, g.ckList_T
    pikList = g.pikList

    gamma_type, sigma_type = g.gamma_type, g.sigma_type

    print(f"Gamma : {g.gamma_type} Sigma : {g.sigma_type}")
    ti = time.time()
    out_ll1, out_rnk1 = g._compute_rnk(T, Y)
    out_rnk1 = np.exp(out_rnk1)
    out_ll1 = out_ll1[:,0]
    print("python ", time.time() - ti)

    N, D = Y.shape
    K, _, Lw = AkList_W.shape
    _, Lt = T.shape

    out_ll2 = np.zeros(N)
    out_rnk2 = np.zeros((N,K))

    tmp_LtLt = np.zeros((Lt, Lt))
    tmp_N = np.zeros(N)
    tmp_N2 = np.zeros(N)
    tmp_ND = np.zeros((N,D))

    tmp_DD = np.zeros((D, D))  # tmp
    tmp_DD2 = np.zeros((D, D))  # tmp

    args = (T, Y, pikList, ckList_T, ckList_W, GammakList_T, GammakList_W, AkList_T, AkList_W,
            bkList, SigmakList, out_rnk2, out_ll2,
            tmp_LtLt, tmp_N, tmp_N2, tmp_ND, tmp_DD, tmp_DD2)

    if gamma_type == "iso":
        if sigma_type == "iso":
            f = Core.cython.gllim.compute_rnk_GIso_SIso
        elif sigma_type == "diag":
            f = Core.cython.gllim.compute_rnk_GIso_SDiag
        elif sigma_type == "full":
            f = Core.cython.gllim.compute_rnk_GIso_SFull
    elif gamma_type == "diag":
        if sigma_type == "iso":
            f = Core.cython.gllim.compute_rnk_GDiag_SIso
        elif sigma_type == "diag":
            f = Core.cython.gllim.compute_rnk_GDiag_SDiag
        elif sigma_type == "full":
            f = Core.cython.gllim.compute_rnk_GDiag_SFull
    elif gamma_type == "full":
        if sigma_type == "iso":
            f = Core.cython.gllim.compute_rnk_GFull_SIso
        elif sigma_type == "diag":
            f = Core.cython.gllim.compute_rnk_GFull_SDiag
        elif sigma_type == "full":
            f = Core.cython.gllim.compute_rnk_GFull_SFull

    ti = time.time()
    f(*args)
    print("cython ", time.time() - ti, "\n")

    assert np.allclose(out_rnk1, out_rnk2), "Rnk"
    assert np.allclose(out_ll1, out_ll2), "LL"




def _check_complet(Lt, Lw, N=2000, D=10, K=40):
    Y = np.random.random_sample((N, D))
    T = np.random.random_sample((N, Lt))

    # g = OldGLLiM(K, Lw, sigma_type="iso", gamma_type="iso")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="diag", gamma_type="iso")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="full", gamma_type="iso")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="iso", gamma_type="diag")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="diag", gamma_type="diag")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="full", gamma_type="diag")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="iso", gamma_type="full")
    # _compare_complet(g, T, Y, Lw)
    # g = OldGLLiM(K, Lw, sigma_type="diag", gamma_type="full")
    # _compare_complet(g, T, Y, Lw)
    g = OldGLLiM(K, Lw, sigma_type="full", gamma_type="full")
    _compare_complet(g, T, Y, Lw)


def _debug():
    _check_complet(2, 5)
    # _check_complet(0, 7)
    # _check_complet(4, 0)
    # _check_Ak(1, 2)
    # _check_Ak(0, 2)
    # _check_Ak(2, 0)


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    _debug()
