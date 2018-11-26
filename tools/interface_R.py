"""Use rpy2 to run xLLiM R-package. Aim to compare R and Python implementations of xLLiM model.
    Shape conventions : in R Package , the K axe is the last. in Python package, the K axe is the first
"""
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects import r, IntVector, ListVector
from rpy2.robjects.packages import importr

from Core.gllim import GLLiM


class RInterface():
    """Hook to R package. Use data from R to run.""" #TODO: Add custom data support ?

    def __init__(self):
        print("Importation du package R...")
        self.xLLiM = importr('xLLiMep')
        print("Importation du package R terminée")

        self.K = 5
        self.maxiter = 100

    @property
    def responses(self):
        return self.dataXllim.rx(IntVector((1,2)),True)

    @property
    def covariates(self):
        return self.dataXllim.rx(IntVector(list(range(3,53))), True)

    @property
    def jointData(self):
        return r['rbind'](self.responses,self.covariates)

    @property
    def npResponses(self):
        return np.array(self.responses).T  # (N,Lt)

    @property
    def npCovariates(self):
        return np.array(self.covariates).T  # (N,D)

    @property
    def npJointData(self):
        return np.array(self.jointData).T # (N,D+Lt)

    @property
    def npTest(self):
        return np.array(self.dataXllimTest).T

    def load_test_data(self):
        r['data']('data.xllim')
        r['data']('data.xllim.test')
        r['data']('data.xllim.trueparameters')
        self.dataXllim = r['data.xllim']
        self.dataXllimTest = r['data.xllim.test']
        self.dataXllimTrueparameters = r['data.xllim.trueparameters']


    def emgm(self):
        r = self.xLLiM.emgm(self.jointData,init=self.K,maxiter=self.maxiter)
        self.r_init = r  #Initialization of parameters
        model = r[1]
        return np.array(model.rx('weight')[0]),\
               np.array(model.rx('mu')[0]).T,\
               np.array(model.rx('Sigma')[0]).transpose((2,1,0)) #Numpy parameters

    def gllim(self,sigma_type,gamma_type,T=None,Y=None,
              Lw=0,in_theta=r('NULL')):
        constraints = {"full":"","iso":"i"}
        c_S = constraints[sigma_type]
        c_G = constraints[gamma_type]
        dic_cst = {"Sigma":c_S}
        if c_G:
            dic_cst["Gammat"] = c_G

        if in_theta:
            in_r =  r('NULL')
            print(np.array(in_theta.rx('c')[0]).shape)
        else:
            in_r = self.r_init

        if T is None:
            T = self.responses
        else:
            T = numpy2ri.numpy2ri(T.T)

        if Y is None:
            Y = self.covariates
        else:
            Y = numpy2ri.numpy2ri(Y.T)

        mod = self.xLLiM.gllim(T,Y,self.K,
                               in_r=in_r,maxiter=self.maxiter,Lw=Lw,
                               cstr=ListVector(dic_cst),
                               in_theta=in_theta,
                               verb=1)
        self.model = mod
        return np.array(mod.rx('pi')[0]),\
               np.array(mod.rx('c')[0]).T,\
               np.array(mod.rx('Gamma')[0]).transpose((2,0,1)),\
               np.array(mod.rx('A')[0]).transpose((2,0,1)),\
               np.array(mod.rx("b")[0]).T,\
               np.array(mod.rx('Sigma')[0]).transpose((2,0,1))

    @staticmethod
    def get_R_theta(pi, c, Gamma, A, b, Sigma):
        """Return a R compatible list from numpy arrays"""
        numpy2ri.activate()
        in_theta = ListVector(dict(
            pi=pi,
            c=c.T,
            Gamma=Gamma.transpose((1,2,0)),
            A = A.transpose((1,2,0)),
            b=b.T,
            Sigma=Sigma.transpose((1,2,0))
        ))
        numpy2ri.deactivate()
        return in_theta

    def glim_inverse_map(self,Y=None):
        if Y is None:
            Y = self.dataXllimTest
        else:
            Y = numpy2ri.numpy2ri(Y.T)
        pred = self.xLLiM.gllim_inverse_map(Y,self.model)
        return np.array(pred.rx('x_exp')[0]).T # N x L

    def test_loggauspdf(self,X,mu,cov):
        numpy2ri.activate()
        r = self.xLLiM.loggausspdf(X,mu,cov)
        numpy2ri.deactivate()
        return r

def is_egal(modele1,modele2):
    def diff(a1,a2):
        return np.max(np.abs(a1 - a2)) / np.max(np.abs(a1))
    print('Diff pi',diff(modele1[0], modele2[0]))
    print('Diff c',diff(modele1[1], modele2[1]))
    print('Diff Gamma',diff(modele1[2], modele2[2]))
    print('Diff A',diff(modele1[3], modele2[3]))
    print('Diff b',diff(modele1[4], modele2[4]))
    print('Diff Sigma',diff(modele1[5], modele2[5]))
    assert np.allclose(modele1[0], modele2[0])
    assert np.allclose(modele1[1], modele2[1])
    assert np.allclose(modele1[2], modele2[2])
    assert np.allclose(modele1[3], modele2[3])
    assert np.allclose(modele1[4], modele2[4])
    assert np.allclose(modele1[5], modele2[5])


def compare_R(sigma_type="iso",gamma_type="iso",Lw=1,K=5):
    ir = RInterface()
    ir.K = K
    ir.load_test_data()
    X = ir.npResponses
    Y = ir.npCovariates
    print(X.shape)
    print(Y.shape)

    X = np.random.multivariate_normal(np.zeros(5)+ 0.2,np.eye(5),200)
    Y = np.random.multivariate_normal(np.zeros(6)+ 10,np.eye(6),200)

    glim = GLLiM(ir.K, Lw, sigma_type=sigma_type, gamma_type=gamma_type, verbose=True)
    glim.init_fit(X,Y,None)



    pi, c, Gamma, A, b, Sigma = glim.pikList,glim.ckList,glim.GammakList,glim.AkList,glim.bkList, glim.full_SigmakList
    print(Gamma.shape)
    sigma_iso = glim.SigmakList
    d = dict(pi=pi, c=c, Gamma=Gamma, A=A, b=b, Sigma=sigma_iso)


    # R run
    in_theta = ir.get_R_theta(pi, c, Gamma, A, b, Sigma)
    ir.maxiter = 10
    mod_R = ir.gllim(sigma_type,gamma_type,in_theta=in_theta,Lw=Lw,T=X,Y=Y)


    # PYthon run
    glim.fit(X, Y,d ,maxIter=ir.maxiter)
    mod_python = (glim.pikList,glim.ckList,glim.GammakList, glim.AkList, glim.bkList, glim.full_SigmakList)

    #Comparaison des modeles
    is_egal(mod_R,mod_python)

    #Comparaison des prédictions
    Y = np.random.random_sample((1000,6))
    pred_R = ir.glim_inverse_map(Y=Y)
    glim.inversion()
    pred_Python = glim.predict_high_low(Y)

    print(np.max(pred_R-pred_Python) / np.max(pred_R))
    assert np.allclose(pred_R,pred_Python)


if __name__ == '__main__':
    ir = RInterface()
    # ir.load_test_data()
    # ir.emgm()
    # m = ir.gllim()
    # pred = ir.glim_inverse_map()
    compare_R("full","iso",Lw=0,K=5)  # OK (4/ 7 /2018)

    # X = np.arange(12).reshape((2,6))
    # mu = np.arange(6) + 0.5
    # cov = 4.7 * np.eye(6)
    # print(ir.test_loggauspdf(X.T,mu[:,None],cov))
    #
    # print(chol_loggausspdf(X.T,mu[:,None],cov))