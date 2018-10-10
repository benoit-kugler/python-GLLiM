"""Implements tools to learn inverse model (bypassing the so-called inversion step).
This approach is only reasonnable if D is small"""
import numpy as np
from scipy.misc import logsumexp

from Core import training
from Core.gllim import GLLiM
from Core.probas_helper import chol_loggausspdf, covariance_melange
from tools.context import HapkeGonio1468_30
from tools.experience import Experience


class InverseGLLiM(GLLiM):
    """Performs standard GLLiM fitting on X,Y, but add method to compute direct density"""

    def inversion(self):
        print("Inversion for InverseGLLiM is useless since we already learned inverse model !")

    def _helper_backward_conditionnal_density(self, X):
        """
        Compute the mean Ak*Y + Bk and the quantities alpha depending of X in (6)
        :param X: shape (L,D)
        :return: mean shape(D,N,K) alpha shape (N,K)
        """
        N = X.shape[0]
        X = X.reshape((N, self.L))

        proj = np.empty((self.D, N, self.K))  # AkS * X + BkS
        logalpha = np.zeros((N, self.K))  # log N(ckS,GammakS)(X)

        for (k, pik, Ak, bk, ck, Gammak) in zip(range(self.K), self.pikList, self.AkList,
                                                self.bkList, self.ckList, self.GammakList):
            proj[:, :, k] = Ak.dot(X.T) + np.expand_dims(bk, axis=1)
            logalpha[:, k] = np.log(pik) + chol_loggausspdf(X.T, ck.reshape((self.L, 1)), Gammak)

        log_density = logsumexp(logalpha, axis=1, keepdims=True)
        logalpha -= log_density
        alpha = np.exp(logalpha)
        return proj, alpha

    def predict_low_high(self, X, with_covariance=False):
        """Backward prediction.
        If with_covariance, returns covariance matrix of the mixture, shape (len(Y),L,L)"""
        N = X.shape[0]
        proj, alpha = self._helper_backward_conditionnal_density(X)
        Ypred = np.sum(alpha.reshape((1, N, self.K)) * proj, axis=2)  # (16)
        if with_covariance:
            covs = np.empty((N, self.D, self.D))
            for n, meann, alphan in zip(range(N), proj, alpha):
                covs[n] = covariance_melange(alphan, meann, self.SigmakList)
            return Ypred.T, covs
        return Ypred.T  # N x L






class LearnInverse(Experience):

    def add_data_training(self,new_data=None,adding_method='threshold',only_added=False,Nadd=None):
        raise NotImplementedError


    def new_train(self,track_theta=False):
        if self.init_local:
            def ck_init_function():
                return self.context.F(self.context.get_X_uniform(self.K))
            gllim = training.init_local(self.Ytrain, self.Xtrain, self.K, ck_init_function, self.init_local, Lw=self.Lw,
                                        sigma_type= self.sigma_type, gamma_type=self.gamma_type,
                                        track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        elif self.multi_init:
            gllim = training.multi_init(self.Ytrain, self.Xtrain, self.K, Lw=self.Lw,
                                        sigma_type= self.sigma_type, gamma_type=self.gamma_type,
                                        track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        else:
            gllim = training.basic_fit(self.Ytrain, self.Xtrain, self.K, Lw=self.Lw,
                                       sigma_type= self.sigma_type, gamma_type=self.gamma_type,
                                       track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        return gllim



if __name__ == '__main__':
    exp = LearnInverse(HapkeGonio1468_30,partiel=(0,1,2,3))
    exp.load_data(regenere_data=False,with_noise=50,N=100000)
    gllim = exp.load_model(100,mode="r",gllim_cls=InverseGLLiM)

    X_predicted = gllim.predict_low_high(exp.Ytest)
    # Normalisation
    X_test = exp.context.normalize_X(exp.Xtest)
    X_predicted = exp.context.normalize_X(X_predicted)
    nrmse = exp.mesures._relative_error(X_predicted, X_test)
    print(np.mean(nrmse),np.median(nrmse),np.std(nrmse),np.max(nrmse))

