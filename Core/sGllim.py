import numpy as np
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters

from Core.gllim import GLLiM, get_full_covariances, jGLLiM


class sGLLiM(GLLiM):

    def coeffecient_randomizationZ(self):
        # return 0.3 / (self.current_iter + 1)
        return 0

    def _draw_contionnalWZ(self,Y,T,coeff_random=0):
        # Sampling of Z
        _ , rnk = self._compute_rnk(Y,T)
        rnk = np.exp(rnk)


        rnk = (rnk + coeff_random)
        rnk /= rnk.sum(axis=1)[:,None]


        N = rnk.shape[0]
        s = rnk.cumsum(axis=1)
        r = np.random.rand(N)[:,None]
        Z = (s < r).sum(axis=1)
        if self.verbose:
            print("Choix Z for first obs:",Z[0])

        I = np.empty(rnk.shape)
        I[:] = np.arange(rnk.shape[1])
        resp = (I == Z[:,None]).astype(float)

        # Sampling of W
        W = np.empty((N, self.Lw))

        if self.Lw == 0:
            return resp , W

        munk, SkList_W = self._compute_rW_Z(Y, T)
        chol_cov = np.linalg.cholesky(SkList_W)
        W = np.random.multivariate_normal(np.zeros(self.Lw),np.eye(self.Lw),N)
        for n,k_n in enumerate(Z):
            mean,c_cov = munk[k_n,n] , chol_cov[k_n]
            W[n] = np.dot(c_cov,W[n]) + mean
        return resp , W

    def threshold(self,resp,n_features):
        card_k = resp.sum(axis=0)
        return (card_k >= (n_features + 1)).prod()
        # return True


    def compute_next_theta(self,T,Y):

        resp ,W = self._draw_contionnalWZ(Y,T,coeff_random=self.coeffecient_randomizationZ())

        # Condition de seuil
        for i in range(5):
            if self.threshold(resp,self.L + self.D):
                break
            resp, W = self._draw_contionnalWZ(Y, T,coeff_random=self.coeffecient_randomizationZ())
            print("Ajustement au seuil")

        return self._gmm_maximization(T,Y,W,resp)

    def _gmm_maximization(self,T,Y,W,resp):
        N = T.shape[0]
        H = np.concatenate((T,W,Y), axis = 1)
        sitype = self.sigma_type == 'iso' and 'spherical' or 'full'
        card_class, m, V = _estimate_gaussian_parameters(H, resp, self.reg_covar,
                                          "full")
        pi = card_class / N

        if self.sigma_type == "iso":
            V = get_full_covariances(V,'spherical',self.K,self.D + self.L)

        dic = jGLLiM.GMM_to_GLLiM(pi, m, V, self.L)
        pi, c, Gamma, A, b, Sigma = dic["pi"], dic["c"], dic["Gamma"], dic["A"], dic["b"], dic["Sigma"]

        if self.sigma_type == 'iso':
            Sigma = np.array([ s[0,0] for s in Sigma])

        ckList_T = c[:,:self.Lt]
        GammakList_T = Gamma[:, :self.Lt, :self.Lt]

        return pi, ckList_T, GammakList_T, A, b, Sigma





class saGLLiM(sGLLiM):

    def temperature(self):
        if self.current_iter <= 10:
            return 1
        return 1 / (self.current_iter - 10 + 2)

    def compute_next_theta(self,T,Y):
        pi1, ckList_T1, GammakList_T1, A1, b1, Sigma1 = GLLiM.compute_next_theta(self, T, Y)
        pi2, ckList_T2, GammakList_T2, A2, b2, Sigma2 = sGLLiM.compute_next_theta(self,T,Y)

        t = self.temperature()

        pi = (1-t) * pi1 + t*pi2
        ckList_T = (1-t) * ckList_T1 + t*ckList_T2
        GammakList_T = (1-t) * GammakList_T1 + t*GammakList_T2
        A = (1-t) * A1 + t*A2
        b = (1-t) * b1 + t*b2
        Sigma = (1-t) * Sigma1 + t*Sigma2
        return pi, ckList_T, GammakList_T, A, b, Sigma

