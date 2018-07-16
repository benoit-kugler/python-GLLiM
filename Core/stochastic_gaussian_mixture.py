"""Gaussian mixture model, with Stochastic EM algorithm."""

import numpy as np
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky

from Core.gllim import MyGMM


class SEMGaussianMixture(MyGMM):
    """Remarque : on utilise la variable Y pour les observations, au lieu de X dans la classe parente."""

    def _compute_Z_conditionnal_density(self,Y):
        """
        Calcule les proba conditionnelles de Z_i sachant Y_i
        :param Y: Observations (n_samples,n_features)
        :return: matrice stochastique (en ligne) (n_samples,n_components)
        """
        proba_cond = np.exp(self._estimate_weighted_log_prob(Y)) # Pi_k * g_k(yi)
        s = proba_cond.sum(axis=1)[:,np.newaxis] # sum_k (Pi_k * g_k(yi))
        return proba_cond / s #On normalise

    def _draw_conditionnal_Z(self,Y):
        """
        Tire un échantillon de loi Z sachant Y

        :param Y: Observations (n_samples, n_features)
        :return: Z (n_samples,n_components) Zik = 1 ssi Zi vaut ek
        """
        M = self._compute_Z_conditionnal_density(Y)
        s = M.cumsum(axis=1)
        r = np.random.rand(M.shape[0])[:,np.newaxis]
        zi = (s < r).sum(axis=1)[:,np.newaxis]
        I = np.empty(M.shape)
        I[:] = np.arange(M.shape[1])
        return (I == zi).astype(float)

    def threshold(self,Z,n_features):
        pik = Z.sum(axis=0)
        return (pik >= (n_features + 1)).prod()

    def _m_step(self, Y, log_resp):
        """M step.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in Y.
        """
        Z = self._draw_conditionnal_Z(Y)
        while not self.threshold(Z,Y.shape[1]): #Condition de seuil
            Z = self._draw_conditionnal_Z(Y)
            print("Ajustement au seuil")

        n_samples, _ = Y.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(Y, Z, self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        self._m_step_callback(Y)

class SAEMGaussianMixture(SEMGaussianMixture):

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        super()._print_verbose_msg_iter_end(n_iter,diff_ll)
        self.current_iter = n_iter + 1 #Prochaine itération

    def _m_step(self, Y, log_resp):
        """M step.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in Y.
        """
        Z = self._draw_conditionnal_Z(Y)
        i = 0
        while i < 10 and not self.threshold(Z, Y.shape[1]):  # Condition de seuil
            Z = self._draw_conditionnal_Z(Y)
            i += 1
            print("Ajustement au seuil")

        n_samples, _ = Y.shape
        SEMweights_, SEMmeans_, SEMcovariances_ = (
            _estimate_gaussian_parameters(Y, Z, self.reg_covar,
                                          self.covariance_type))
        SEMweights_ /= n_samples

        EMweights_, EMmeans_, EMcovariances_ = (
            _estimate_gaussian_parameters(Y, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        EMweights_ /= n_samples

        r = self.current_iter
        gr = self.gamma(r)
        self.means_ = (1 - gr) * EMmeans_ + gr * SEMmeans_
        self.weights_ = (1 - gr) * EMweights_ + gr * SEMweights_
        self.covariances_ = (1 - gr) * EMcovariances_ + gr * SEMcovariances_

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        self._m_step_callback(Y)

    @staticmethod
    def gamma(r):
        return 1 / np.sqrt( r + 1)

