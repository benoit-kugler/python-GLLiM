# Ce module permet de simuler des donneés qui peuvent être utilisées pour tester différents algorithmes.
import logging
import os
import random

import numpy as np
import pyDOE
import rpy2.robjects.packages
import scipy.io
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from rpy2 import robjects

from hapke.hapke_vect_opt import Hapke_vect
from hapke.cython import Hapke_cython

randtoolbox = robjects.packages.importr('randtoolbox')


def test_random(u):
    tol = 10**-16
    if (u+tol>1.).any() or (u-tol<0.).any():
        logging.debug("Generation SOBOL -> ERROR: outside 0 1 ! ")
        return True
    else:
        return False

def random_sequence_rqmc(dim, init=True, n=1):
    """
    generates RQMC random sequence
    """
    dim = np.int(dim)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    u = np.array(randtoolbox.sobol(n=n, dim=dim, init=init, scrambling=1, seed=random_seed)).reshape((n,dim))
    # randtoolbox for sobol sequence
    while test_random(u):
        random_seed = random.randrange(10**9)
        u = np.array(randtoolbox.sobol(n=n, dim=dim, init=init, scrambling=1, seed=random_seed)).reshape((n,dim))

    return(u)


def _xlims_to_P(xlims):
    """Return latex code for definition domain defined by xlims"""
    return "\\times".join(f""" \left[ {x[0]} , {x[1]} \\right] """ for x in xlims)


class abstractFunctionModel:
    """Implements base methods to work with a model Y = F(X) + E"""

    PARAMETERS = None
    """Labels of X variables"""

    XLIMS = None
    """np.array defining maximum values of X"""

    DEFAULT_VALUES = None
    """x default values"""

    LABEL = None
    """String describing the context (latex formatting)"""

    DESCRIPTION = "Generic function $F$"

    D = None
    """Dimension of F values"""

    PREFERED_MODAL_PRED = "prop"
    """Default method to find modes (int or float or "prop") """

    def __new__(cls, *args, **kwargs):
        if cls.LABEL is None:
            raise NotImplementedError(f"Class atribut LABEL should be set for class {cls.__name__}")
        return super().__new__(cls)

    def __init__(self, partiel=None):
        """partiel is a tuple of indexes of x variable to take in account. None to take all variables"""
        self.partiel = partiel

    def F(self, X, check=False):
        """Returns F(X). If partiel, X is only the partial values"""
        X = self._prepare_X(X)
        Y = self._F(X)
        assert (not check) or np.isfinite(Y).all()
        return Y

    def _prepare_X(self, X):
        return self.to_X_physique(X)

    def _F(self, X):
        raise NotImplementedError

    @property
    def variables_lims(self):
        if self.partiel:
            var_lims = self.XLIMS[[*self.partiel]]
        else:
            var_lims = self.XLIMS
        return var_lims

    @property
    def variables_range(self):
        xlims = self.variables_lims
        return xlims[:, 1] - xlims[:, 0]

    @property
    def variables_names(self):
        if self.partiel:
            var_names = self.PARAMETERS[[*self.partiel]]
        else:
            var_names = self.PARAMETERS
        return var_names

    @classmethod
    def _get_L(cls,partiel):
        return len(partiel) if (partiel is not None) else len(cls.XLIMS)

    @property
    def L(self):
        return self._get_L(self.partiel)


    def normalize_X(self,X):
        """Takes X values and returns a version in [0,1]"""
        return (X - self.variables_lims[:,0]) / (self.variables_range)

    def to_X_physique(self, X):
        """Maps mathematical X valued to physical ones"""
        return self.variables_lims[:, 0] + self.variables_range * X

    def to_Cov_physique(self, Cov):
        """Return tA C A """
        A = np.diag(self.variables_range)
        return A.T * Cov * A

    def normalize_Y(self, Y):
        """Should return Y version in [0,1]. Used only in measures."""
        return Y

    def add_noise_data(self, Y, covariance=None, mean=None):
        """Gaussian Noise
        covariance may be an homotéthie (float), diagonal (1D array), full (2D array)"""
        Y = np.copy(Y)
        N, D = Y.shape
        if mean is None:
            mean = np.zeros(D)
        elif (type(mean) is float) or (type(mean) is int):
            mean = mean * np.ones(D)

        if (type(covariance) is float) or (type(covariance) is int):
            std = np.sqrt(covariance) * np.eye(D)
        elif covariance is None:
            std = np.zeros((D, D))
        elif covariance.ndim == 1:
            std = np.diag(np.sqrt(covariance))
        else:
            std = np.linalg.cholesky(covariance)
        noise = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=N)
        return Y + std.dot(noise.T).T + mean[None, :]


    def get_X_sampling(self, N, method ='sobol'):
        """Uniform random generation in [0,1] Suitable to learn Hapke.
        method is one of 'random' 'latin' 'sobol'
        If partiel, returns partial samples."""

        xlims = self.variables_lims

        if method == "latin":
            alea = pyDOE.lhs(len(xlims),samples=N)
        elif method == "sobol":
            alea = random_sequence_rqmc(len(xlims),n=N)
        else:
            alea = np.random.random_sample((N,len(xlims)))

        return alea

    def get_data_training(self,N,method="sobol"):
        """Returns training sample X , Y of size N with synthetic X and Y = Hapke(X)
        X shape : (N_train , len(partiel))
        Y_shape : (N_train , len(geometries))"""
        alea = self.get_X_sampling(N, method=method)
        Y = self.F(alea)
        return alea, Y


    def _get_X_grid(self,N):
        x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
        variable = np.array([x.flatten(), y.flatten()]).T
        return variable

    def Fsample(self, N, cov_noise=None, mean_noise=None):
        """Return value of F over a grid.
        If partiel, other components are fixed to default values.
        Returns
            x : shape(N,N)
            y : shape(N,N)
            H : shape(D,N,N) (compent by component)
        """
        if not (self.partiel and len(self.partiel) == 2 or len(self.XLIMS) == 2):
            raise ValueError("Fsample expects only 2 variables context")

        X = self._get_X_grid(N)
        Y = self.F(X)

        if (cov_noise is not None) or (mean_noise is not None):
            Y = self.add_noise_data(Y, covariance=cov_noise, mean=mean_noise)

        H = np.array([y.reshape((N, N)) for y in Y.T])

        x = X[:, 0].reshape((N, N))
        y = X[:, 1].reshape((N, N))

        return x, y, H

    def get_X_uniform(self, K):
        """Returns a uniform grid with K points around the X space"""
        return self.get_X_sampling(K,method="sobol")


    def is_X_valid(self,X):
        """Returns a mask of theoretically correct values"""
        if type(X) is list:
            mask = [(np.all((0 <= x) * (x <= 1), axis=1) if x.shape[0] > 0 else None) for x in X]
        else:
            mask = np.array([np.all((0 <= x) * (x <= 1)) for x in X])
        return mask

    def is_Y_valid(self,Y):
        """By default,accepts all values"""
        mask = np.ones(Y.shape[0], dtype=bool)
        return mask

    def _dF_finite(self, X):
        eps = 0.000001
        FX = self.F(X)
        dFs = np.empty((X.shape[0],self.D,self.L))
        for i in range(self.L):
            ei = np.arange(self.L) == i
            FXi = self.F(X + eps * ei)
            dFs[:,:,i] = (FXi - FX) / eps
        return dFs

    def dF(self,X):
        """Using finite difference to compute dF. Other methods like automatic diff or symbolic diff can be used"""
        return self._dF_finite(X)




class abstractSimpleFunctionModel(abstractFunctionModel):

    D = 1

    DEFAULT_VALUES = np.array([0.5])

    XLIMS = np.array([[0, 1]])

    PARAMETERS = np.array(["x"])

    YLIMS = np.array([[0, 1]])

    def normalize_Y(self, Y):
        return (Y - self.YLIMS[:, 0]) / (self.YLIMS[:, 1] - self.YLIMS[:, 0])


class SurfaceFunction(abstractSimpleFunctionModel):
    DEFAULT_VALUES = np.array([0.5, 0.5])

    XLIMS = np.array([[0, 1], [0, 1]])

    PARAMETERS = np.array(["$x_1$", "$x_2$"])

    LABEL = "$x_1 ,x_2 \mapsto 1 - \\frac{x_1^2 + x_2^3}{2}$ "

    def _F(self, X):
        return (-(X[:, 0:1] ** 2 + X[:, 1:2] ** 3) / 2) + 1

    def Fcoupe(self, z, Y):
        """return X such as F(X,Y) = z"""
        arg = 2 * (1 - z) - Y ** 3
        sign = (arg >= 0) * 2 - 1
        return np.sqrt(arg)

class SquaredFunction(abstractSimpleFunctionModel):
    LABEL = "Fonction carrée"

    DESCRIPTION = r"$x \rightarrow x^{2}$, sur $[0,1]$"

    YLIMS = np.array([[0, 0.25]])

    def _F(self, X):
        return np.square(X - 0.5)


class WaveFunction(abstractSimpleFunctionModel):
    DESCRIPTION = f"  $x \mapsto \cos(10x) $, sur ${ _xlims_to_P(abstractSimpleFunctionModel.XLIMS) } $"
    LABEL = "Fonction $\cos$"

    YLIMS = np.array([[0, 1]])

    def _F(self, X):
        return np.cos(X * 30)

    def dF(self,X):
        return (-30 * np.sin(30 * X))[:, :, None]

    def is_Y_valid(self,Y):
        return np.array([np.all((-1 <= y) * (y <= 1)) for y in Y])

class MixedFunction(abstractSimpleFunctionModel):

    def _F(self, X):
        return (X >= 1) * np.cos(X * 10) + (X <= 1) * X


class TwoSolutionsFunction(abstractSimpleFunctionModel):
    D = 2
    DEFAULT_VALUES = np.array([0.5] * 2)
    XLIMS = np.array([[0, 1]] * 2)
    YLIMS = np.exp([[-1, 2]] * 2)
    PARAMETERS = np.array(["x{}".format(i + 1) for i in range(2)])

    LABEL = "Fourche"
    DESCRIPTION = "$x \mapsto ( (x_{1} - 0.5)^2 , x_{2} , ... , x_{L})$," + f" définie sur ${ _xlims_to_P(XLIMS) }$." \
                  + " Fonction ayant deux antécédants pour chaque $y$."

    def _F(self, X):
        out = np.empty((X.shape[0], self.L))
        out[:, 0] = (X[:, 0] - 0.5) ** 2
        out[:, 1:self.L] = X[:, 1:self.L]
        return out

class abstractExpFunction(abstractSimpleFunctionModel):

    MAX_F = np.exp(2)

    COERCIVITE_F = 1

    LIPSCHITZ_F = np.exp(1)

    LABEL = "Fonction $\exp$"

    DESCRIPTION = "$(x_{{i}}) \mapsto (\exp(x_{{i}}) )$, sur ${}$"

    PREFERED_MODAL_PRED = 1

    YLIMS = np.array([[1, np.exp(1)]])

    def _F(self, X):
        return np.exp(X)


def InjectiveFunction(d, **kwargs) -> type:
    xlims = np.array([[-1, 2]] * d)
    description = abstractExpFunction.DESCRIPTION.format(_xlims_to_P(xlims))
    attrs = dict(D=d,
                 DEFAULT_VALUES=np.array([0.5] * d),
                 XLIMS=xlims,
                 YLIMS=np.exp([[-1, 2]] * d),
                 DESCRIPTION=description,
                 PARAMETERS=np.array(["x{}".format(i + 1) for i in range(d)]), **kwargs)

    return type("InjectiveFunction{}".format(d), (abstractExpFunction,), attrs)


class ExampleFunction(abstractSimpleFunctionModel):
    D = 1
    DEFAULT_VALUES = np.array([0.5] * 2)
    XLIMS = np.array([[0, 1]] * 2)
    YLIMS = np.exp([[-1, 2]] * 4)
    PARAMETERS = np.array(["$x_{1}$", "$x_{2}$"])

    LABEL = "Surface"

    DESCRIPTION = f"$(x,y) \mapsto x_{1}^2 + x_{2}^3$, sur ${ _xlims_to_P(XLIMS) }$"

    def _F(self, X):
        return X[:, 0:1] ** 2 + X[:, 1:2] ** 3


class abstractHapkeModel(abstractFunctionModel):
    """Adds geometries support"""

    BASE_PATH = "../DATA/HAPKE"

    SCATTERING_VARIANT = "2002"

    """Label of variables"""
    PARAMETERS = np.array(['$\omega$', r'$\overline{\theta}$', '$b$', '$c$', '$H$', '$B_{0}$'])

    """Therorical intervals [0,xmax]"""
    XLIMS = np.array([[0, 1], [0, 31], [-0.05, 1], [-0.05, 1], [-0.05, 1.2], [-0.05, 1.2]])

    DEFAULT_VALUES = np.array([0.5,15,0.5,0.5,0.5,0.5])
    """Mean default values"""

    PREFERED_MODAL_PRED = 2

    LABEL = "$F_{hapke}$"

    def __init__(self, partiel=None):
        super().__init__(partiel)
        self.geometries = None # emergence, incidence, azimuth
        self._load_context_data()

        self.path_dF = os.path.join(self.BASE_PATH,self.__class__.__name__ + "_dF.dill")
        # self._load_dF() Buggy. Use finite difference for now
        logging.debug("Uses finite difference for dF computing.")

    @property
    def D(self):
        return self.geometries.shape[2]

    def _load_context_data(self):
        """Setup context to be able to compute F(X).
        Here, needs to set up geometries as array of shape (3,1 _ )"""
        pass


    def _genere_data_for_Hapke(self, X):
        """Renvoi un tableau construit à partir des N réalisations X des paramètres physiques et des N_geom géométries.
        Ce tableau est destiné à la fonction Hapke, dont le résultat doit être re-coupé.
        Return shape : (dim(X) + dim(Geom), N*N_geom )
        """
        # Configurations géométriques
        t,t0,p = self.geometries
        N_geom = t.shape[1]
        dim_X = X.shape[1]
        l = []
        for xi in X:
            xi_dupl = np.ones((N_geom,dim_X)) * xi
            m  = np.concatenate((t0.T,t.T,p.T,xi_dupl),axis=1)
            l.append(m)
        final = np.concatenate(l,axis=0)
        return final.T  #Pour isoler chaque composant pour Hapke

    def _prepare_X(self, X):
        """If partiel, other components are fixed to default values."""
        X = self.to_X_physique(X)
        if self.partiel:
            N = X.shape[0]
            if N == 0:
                return np.empty(X.shape)
            Xfull = np.array([ self.DEFAULT_VALUES ]  * N)
            Xfull[:,self.partiel] = X
        else:
            Xfull = X
        return Xfull

    def F(self, X, check=False):
        """If partiel, other components are fixed to default values."""
        Xfull = self._prepare_X(X)
        t, t0, p = self.geometries
        args = (np.array(Xfull[:, i], dtype=np.double) for i in range(Xfull.shape[1]))
        Y = Hapke_cython(np.array(t0[0], dtype=np.double), np.array(t[0], dtype=np.double),
                         np.array(p[0], dtype=np.double), *args)
        assert (not check) or np.isfinite(Y).all()
        return Y


    def compute_dF(self):
        """Compute dF through symbolic calculus, and save the result.
        Long to execute, do not repeat unless geometries have changed."""
        from hapke import hapke_sym
        hapke_sym.save_dF(self, savepath=self.path_dF)
        logging.info("dF function saved in {}".format(self.path_dF))

    def _load_dF(self):
        try:
            from hapke import hapke_sym
            dF = hapke_sym.load_dF(self.path_dF)
        except FileNotFoundError:
            logging.warning("Warning ! No dF found. Use compute_dF once to set it up, "
                  "or you won't be able to use dGLLiM (GLLiM version using F gradient) ")
        except EOFError:
            logging.warning("Failed to load dF !")
        else:
            self.dF_total = dF
            logging.info("dF loaded from {}".format(self.path_dF))


    def dF(self,X,permutation=None):
        """X might be partiel"""
        # Xfull = self._prepare_X(X)
        # dF = self.dF_total(Xfull)
        # if permutation is not None: # permutation of variables
        #     dF = dF[:,:,permutation]
        # if self.partiel:
        #     dF = dF[:,:,self.partiel]
        # return dF
        return self._dF_finite(X)


    def _test_dF(self):
        """Debug purpose. Compare dF with finite difference"""
        x0 = self.get_X_sampling(1)
        eps = 0.00000001
        h = np.arange(len(self.variables_names))
        y = (self.F(x0 + eps * h) - self.F(x0)) / eps
        logging.debug(y[0] - self.dF(x0)[0].dot(h))

    def normalize_Y(self, Y):
        """Does nothing since Y is already (almost) in [0,1]"""
        return Y



class HapkeContext(abstractHapkeModel):
    """Used by Experience. Defines meta data of the study"""

    DESCRIPTION = "Contexte d'une observation satellitaire CRISM, du site d’atterrissage du rover MER-Opportunity " \
                  u"à Meridiani Planum (MARS)."
    LABEL = "$F_{hapke}$ CRISM"

    DEFAULT_VALUES = np.array([0.5, 15, 0.5, 0.5, 0.5, 0])

    EXPERIENCES = ["exp1/Inv_FRT193AB_S_Wfix_0.33_rho_mod.mat",
                 "exp2/Inv_FRT0A941_S_Wfix_0.12508_rho_mod.mat"]

    RESULT_MEAN_INDEXS = np.array([11, 8, 2, 5, 14, 17])
    RESULT_STD_INDEXS = np.array([12, 9, 3, 6, 15, 18])

    def __init__(self, partiel=None, index_exp=0):
        self.index_exp = index_exp
        super().__init__(partiel)

    def _load_context_data(self):
        chemin = os.path.join(self.BASE_PATH,self.EXPERIENCES[self.index_exp])
        d = scipy.io.loadmat(chemin)
        self.geometries = np.array([d["theta0"], d["theta"], d["phi"]])


    def get_observations(self,wave_index=0):
        "Renvoi les observations"
        chemin = os.path.join(self.BASE_PATH,self.EXPERIENCES[self.index_exp])
        d = scipy.io.loadmat(chemin)
        obs = d["cub_rho_mod"][:,:,wave_index]
        return obs

    def get_spatial_coord(self):
        assert self.index_exp == 0
        chemin = os.path.join(self.BASE_PATH,"exp1/FRT000193AB_result.mat")
        d = scipy.io.loadmat(chemin)
        a = np.array(d['result_summary'])
        latlong = a[:,(0,1),0]
        #cleaning
        mask = [-90 <= x[0] <= 90 and -180 <= x[1] <= 180 for x in latlong]
        latlong = latlong[mask]
        return latlong , mask

    def get_result(self,full=False,with_std=False):
        assert self.index_exp == 0
        chemin = os.path.join(self.BASE_PATH,"exp1/FRT000193AB_result.mat")
        d = scipy.io.loadmat(chemin)
        a = np.array(d['result_summary'])
        if full:
            return a[:,:,0]
        if self.partiel:
            index = self.RESULT_MEAN_INDEXS[[*self.partiel]]
            index_std = self.RESULT_STD_INDEXS[[*self.partiel]]
        else:
            index = self.RESULT_MEAN_INDEXS
            index_std = self.RESULT_STD_INDEXS
        if with_std:
            return a[:,index,0] , a[:,index_std,0]
        return a[:,index,0]


class HapkeContext1993(HapkeContext):

    SCATTERING_VARIANT = "1993"

    DESCRIPTION = HapkeContext.DESCRIPTION + " - Variante 1993"


class abstractHapkeGonio(abstractHapkeModel):

    EXPERIENCES = [1468,1521,"JSC1"]

    MODEL_PATH =  "BRDF_{}_c2_filtered.mat"

    EXP_NAME = None

    GEOMETRIES = None

    LABEL = "$F_{hapke}$ Gonio."

    DESCRIPTION = "Configuration de mesure de réflectance en laboratoire"

    def _load_context_data(self):
        exp = self.EXP_NAME
        chemin = os.path.join(self.BASE_PATH,self.MODEL_PATH.format(exp))
        d = scipy.io.loadmat(chemin)
        angles = d["angles_filtered_{}".format(exp)]
        geom = angles[0:3,:].T
        mask = [ (x != 0).all() for x in geom]
        self.geometries = geom[mask].T[:,None,:]
        # cleaning of unsafe geometries, id with zero

        if self.GEOMETRIES is not None:
            self.geometries = self.geometries[:,:,self.GEOMETRIES]

    def get_observations(self):
        exp = self.EXP_NAME
        chemin = os.path.join(self.BASE_PATH,self.MODEL_PATH.format(exp))
        d = scipy.io.loadmat(chemin)
        y = d["data_filtered_{}".format(exp)]
        if self.GEOMETRIES is not None:
            y = y[:,self.GEOMETRIES]
        return y



class HapkeGonio1468(abstractHapkeGonio):
    EXP_NAME = 1468

    LABEL = abstractHapkeGonio.LABEL + " C1"
    DESCRIPTION = abstractHapkeGonio.DESCRIPTION + " - sur échantillon Charbon 1468."

class HapkeGonio1521(abstractHapkeGonio):
    EXP_NAME = 1521

    LABEL = abstractHapkeGonio.LABEL + " C2"
    DESCRIPTION = abstractHapkeGonio.DESCRIPTION + " - sur échantillon Charbon 1521."


class HapkeGonioJSC1(abstractHapkeGonio):
    EXP_NAME = "JSC1"

    LABEL = abstractHapkeGonio.LABEL + " AM"
    DESCRIPTION = abstractHapkeGonio.DESCRIPTION + " - sur échantillon Analogue martien JSC1."

class HapkeGonio1468_30(HapkeGonio1468):
    GEOMETRIES = (np.arange(105) % 3) == 0


class HapkeGonio1468_50(HapkeGonio1468):
    GEOMETRIES = (np.arange(105) % 2) == 0


class abstractLabContext(abstractHapkeModel):
    """Uses lab setup"""

    HAPKE_VECT_PERMUTATION = [2, 3, 1, 0, 5, 4]
    """Index of parameters in reference given by Hapke_vect"""

    """Label of variables"""
    PARAMETERS = abstractHapkeModel.PARAMETERS[HAPKE_VECT_PERMUTATION]

    """Therorical intervals [0,xmax]"""
    XLIMS = np.array([ [-0.05, 1], [-0.05, 1],[0, 31],[-0.05, 1],  [-0.05, 1.2], [-0.05, 1.2]])

    DEFAULT_VALUES = np.array([0.5,0.5,15,0.5,0.5,0.5])
    """Mean default values"""

    BASE_PATH = "../DATA/HAPKE/resultats_photom_lab_pilorget_2016/"

    EXPERIENCES = [("lab_data/data_nontronite_ng1_corr.sav","result_photom/NG1_new_optim_100iter","brf_nontronite"),
                   ("lab_data/data_olivine_olv_corr.sav","result_photom/OLV_new_optim_100iter","brf_olv")]

    PDF_NAMES = ["b","c","theta","omega","",""]

    def __init__(self,partiel,exp_index):

        exp_data = self.EXPERIENCES[exp_index]
        self.data_path = os.path.join(self.BASE_PATH,exp_data[0])
        super().__init__(partiel)
        self.result_path = os.path.join(self.BASE_PATH,exp_data[1])
        self.brdf_field = exp_data[2]

        # Results index
        filename = os.path.join(self.result_path, "optim_list.lis")
        with open(filename) as f:
            lines = [s.strip() for s in f.readlines()]
        self.result_files_index = lines


    def _load_context_data(self):
        d = scipy.io.readsav(self.data_path)
        self.geometries = np.array([d["inc"][None,:], d["eme"][None,:], d["azi"][None,:]])
        self.wavelengths = d["wave_value"]

    def _prepare_X(self, X):
        """Changes X variable order to match Hapke_vect"""
        Xfull = super()._prepare_X(X)
        i = np.argsort(self.HAPKE_VECT_PERMUTATION)
        return Xfull[:,i]

    def dF(self,x):
        return super().dF(x,self.HAPKE_VECT_PERMUTATION)

    def get_observations(self):
        d = scipy.io.readsav(self.data_path)
        return d[self.brdf_field]

    def get_result(self,index_wavelength=None,full=False):
        files = index_wavelength and [self.result_files_index[index_wavelength]] or self.result_files_index
        r = []
        covs = []
        for f in files:
            filename = os.path.join(self.result_path,f)
            d = scipy.io.readsav(filename)
            if full:
                r.append(d)
            else:
                r.append(d["estim_m"])


            covs.append(d["var_m"])

        r = np.array(r)
        covs = np.array(covs)

        if self.partiel:
            r = r[:,self.partiel]
            covs = covs[:,self.partiel]

        if index_wavelength is not None: #File already choosen
            r = r[0]
            covs = covs[0]

        return r, covs



    def get_images_path_densities(self,index):
        ind = self.partiel and self.partiel[index] or index
        return [os.path.join(self.result_path,"pdf_plot","{0:03}_pdf_{1}.png".format(i,self.PDF_NAMES[ind]))
                for i in range(len(self.result_files_index))]

    LABEL = "Config. Wavelengths"

    DESCRIPTION = "Configuration de mesure de réflectance en laboratoire (divers longueurs d'ondes)"

class LabContextNontronite(abstractLabContext):
    LABEL = "$F_{hapke}$ Nontronite"

    def __init__(self, partiel=None):
        super().__init__(partiel,0)

    DESCRIPTION = abstractLabContext.DESCRIPTION + " - sur échantillon de nontronite."


class LabContextOlivine(abstractLabContext):
    LABEL = "$F_{hapke}$ Olivine"

    DESCRIPTION = abstractLabContext.DESCRIPTION + " - sur échantillon d'olivine."


    COERCIVITE_F = 1
    LIPSCHITZ_F = 1

    def __init__(self, partiel=None):
        super().__init__(partiel,1)


class MergedLabObservations(LabContextOlivine):

    def get_observations(self):
        obs1 = super(MergedLabObservations, self).get_observations()
        obs2 = LabContextNontronite().get_observations()
        return np.concatenate((obs1, obs2), axis=0)


class abstractGlaceContext(abstractHapkeModel):

    BASE_PATH = "../DATA/HAPKE/glace"

    EXPERIENCE = ""


    @staticmethod
    def _clean_NAN(Y):
        mask = np.array([(~ np.isnan(y)).all() for y in Y])
        return mask

    @staticmethod
    def _clean_spatial_coord(S):
        """Removes entry with near 0 coordinates"""
        mask = np.array([ not np.allclose(s,0) for s in S])
        return mask

    def _load_context_data(self):
        chemin = os.path.join(self.BASE_PATH,self.EXPERIENCE)
        d = scipy.io.loadmat(chemin)
        self.__data = d  # cached
        self.geometries = np.array([d["theta0"], d["theta"], d["phi"]])
        self.wave_lengths = np.array(d["lambda"])[:,0]

    def get_observations_fixed_wl(self,wave_index=0):
        "Renvoi les observations pour une longueur d'onde donnée"
        d = self.__data
        obs = d["cub_rho_mod"][:,:,wave_index]
        return obs , self._clean_NAN(obs)


    def get_observations_fixed_coord(self,coord_index=0):
        """Renvoi un jeu d'observation pour une coordonnée spatial donnée"""
        d = self.__data
        data = d["cub_rho_mod"]
        obs = data[coord_index,:,:].T
        return obs , self._clean_NAN(obs)


    def get_spatial_coord(self):
        """Renvoi les coordonnées spatiales.
        N'applique pas de masque"""
        d = self.__data
        latlong = d["cub_geo_mod"][:,6,(2,1)]
        return latlong , self._clean_spatial_coord(latlong)


class VoieS(abstractGlaceContext):

    EXPERIENCE = "Inv_FRT144E9_S_Wfix_0.60427_rho_mod.mat"

    LABEL = "Config. Glace - Voie S"


class VoieL(abstractGlaceContext):
    EXPERIENCE = "Inv_FRT144E9_L_Wfix_0.61075_rho_mod.mat"

    LABEL = "Config. Glace - Voie L"


# ------------- Dummy Linear Injective F ------------- #
class LinearFunction(abstractFunctionModel):
    F_matrix = np.diag(1 + 0.5 * np.arange(10))[:, :4]

    D = 10

    PRIOR_COV = 0.2 * np.eye(4)

    LABEL = "Fonction linéaire injective"

    def _F(self, X):
        return self.F_matrix.dot(X.T).T

    def get_X_sampling(self, N, method='sobol'):
        return np.random.multivariate_normal(np.zeros(4), self.PRIOR_COV, size=N)

    def normalize_X(self, X):
        """Takes X values and returns a version in [0,1]"""
        return X

    def to_X_physique(self, X):
        """Maps mathematical X valued to physical ones"""
        return X




if __name__ == '__main__':
    def test_bruit(N=10000):
        h = HapkeContext(None)
        X, Y = h.get_data_training(N)
        Y = h.add_noise_data(Y, 0.02, 0.2)
        FX = h.F(X)
        mu_hat = (Y - FX).sum(axis=0) / N
        diff = Y - FX - mu_hat
        sigma_hat = np.array([d[:, None].dot(d[None, :]) for d in diff]).sum(axis=0) * (N - 1) / N
        print(mu_hat)
        print(sigma_hat)


    # test_bruit()


    # h = abstractExpFunction(None)
    # X = h.get_X_sampling(10000)
    # print(X.min(),X.max())
    # Y = h.get_X_sampling(10000)
    # taux = np.abs( (h.F(X) - h.F(Y)) / (X - Y))
    # print(taux.min())
    # assert np.all( taux >= h.COERCIVITE_F)
    # pyplot.hist(taux,np.linspace(0,6,100))
    # pyplot.axvline(h.COERCIVITE_F)
    # pyplot.show()

    # h = VoieS(None)
    # Y , mask = h.get_observations_fixed_wl()
    # print(Y.shape)
    # # print(h.get_spatial_co(xord()[mask])1
    # h = HapkeGonio1468_30(partiel=(0,1,2,3))
    # print(h.geometries.shape)
    # h = InjectiveFunction(10)(None)
    # print(h.D,h.L)
    # h.compute_dF()
    # h._test_dF()
    #
    # X = h.get_X_sampling(300)
    # h = HapkeContext((0, 1))
    # x, y, Z = h.Fsample(200, cov_noise=np.arange(h.D) + 1, mean_noise=4.)
    # axe = pyplot.subplot(projection="3d")
    # axe.plot_surface(x, y, Z[0], color="gray", alpha=0.4, label="True F")
    # pyplot.show()

    # h = LabContextNontronite()
    # # print(h.get_observations().shape)
    # # print(h.geometries)
    # # h = LabContextOlivine()
    # y = h.get_observations()
    # print(h.geometries.shape)
    # np.savetxt("/home/bkugler/Documents/reunion8_11/olivine_geom.txt",h.geometries[:,0,:],fmt="%.2f")
    # pyplot.plot(h.wavelengths, y)
    # pyplot.savefig("/home/bkugler/Documents/reunion8_11/nontronite.png")
    #
    # h = LabContextOlivine()
    # y = h.get_observations()
    # pyplot.clf()
    # pyplot.plot(h.wavelengths, y)
    # pyplot.savefig("/home/bkugler/Documents/reunion8_11/olivine.png")
    # print(h.geometries)

    print(LinearFunction.F_matrix)
