import numpy as np
from scipy.special._ufuncs import expit, logit

from tools.context import LabContextOlivine
from tools.experience import SecondLearning


class LogisticOlivineContext(LabContextOlivine):
    LABEL = LabContextOlivine.LABEL + " - logit"
    DESCRIPTION = LabContextOlivine.DESCRIPTION + " - Avec transformation logistique"

    def get_X_sampling(self, N, method ='sobol'):
        X = super().get_X_sampling(N,method)
        X = logit( (X - self.variables_lims[:,0]) / (self.variables_range) )
        return X

    def _prepare_X(self, X):
        """X = expit( x-a / b-a )"""
        X = self.to_X_physique(X)
        return super()._prepare_X(X)

    def _get_X_grid(self,N):
        X = super()._get_X_grid(N)
        return logit( (X - self.variables_lims[:,0]) / (self.variables_range) )

    # Since we use finite difference, dF is already OK
    # def dF(self, X):
    #     dF = super().dF(X) # dF_L(x) since _prepare_X is called
    #     emx = np.exp(-X)
    #     D = self.variables_range * emx / (1 + emx)**2
    #     return np.matmul(dF,np.array([np.diag(d) for d in D]))

    def is_X_valid(self,X : np.array):
        if type(X) is list:
            mask = [(np.ones(x.shape[0],dtype=bool) if x.shape[0] > 0 else None) for x in X]
        else:
            mask = np.ones(X.shape[0],dtype=bool)
        return mask

    def normalize_X(self,X):
        return expit(X)

    def to_X_physique(self,X):
        """Maps mathematical X valued to physical ones"""
        return self.variables_lims[:,0] + self.variables_range * expit(X)



# h._test_dF()  OK ( 3/8/2018 )





def compare_MCMC():
    index = 1
    MCMC_X, Std = exp.context.get_result()
    exp.mesures.plot_density_sequence(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                                      index=index, Xref=MCMC_X, StdRef=Std, with_pdf_images=True, varlims=(-0.2, 1.2),
                                      regul="exclu",post_processing=exp.context.to_X_physique)

def main():
    exp.mesures.plot_mesures(gllim)

if __name__ == '__main__':
    exp = SecondLearning(LogisticOlivineContext, partiel=(0, 1, 2, 3))
    # exp.load_data(regenere_data=False, with_noise=50, N=10000, method="sobol")
    # dGLLiM.dF_hook = exp.context.dF
    # # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    # gllim = exp.load_model(1000, mode="l", with_GMM=False, track_theta=False, init_local=500,
    #                        sigma_type="iso", gamma_type="full", gllim_cls=dGLLiM)
    # compare_MCMC()
    h = LogisticOlivineContext(None)
    h._test_dF()