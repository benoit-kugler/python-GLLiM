"""Tests a deterministic relation beetween B and C"""

import numpy as np

from Core.gllim import GLLiM
from tools.context import HapkeContext, LabContextOlivine, abstractHapkeModel
from tools.experience import SecondLearning


def Crelation(B):
    C_prime =  3.29 * np.exp(-17.4 * (B ** 2)) - 0.908
    return (C_prime + 1) / 2


def dCrelation(B):
    0.5 * 3.29 * (- 2 * 17.4 * B) * np.exp()

class HapkeCRelationContext(HapkeContext):
    """Suppose a relation c = F(b). c is not learned"""

    """Label of variables"""
    PARAMETERS = np.array(['\omega', r'\overline{\theta}', 'b', 'H', 'B_{0}'])

    """Therorical intervals [0,xmax]"""
    XLIMS = np.array([[-0.05,1],[0,31], [-0.05,1], [-0.05,1.2], [-0.05,1.2]])

    DEFAULT_VALUES = np.array([0.5,15,0.5,0.5,0.5])
    """Mean default values"""

    DESCRIPTION = HapkeContext.DESCRIPTION + " Modèle de Hapke à trois paramètres, " \
                                             "où $c$ est déduit de $b$ par $ c = \mathcal{R}(b) $"
    LABEL = HapkeContext.LABEL + " -$\mathcal{R}$"

    def _prepare_X(self, X):
        Xfull = super()._prepare_X(X)
        C =  Crelation(Xfull[:,2])
        Xfull = np.concatenate((Xfull[:,0:3],C[:,None],Xfull[:,3:]),axis=1)
        return Xfull


class LabContextOlivine_C(LabContextOlivine):

    """Label of variables"""
    PARAMETERS = np.array(["b",r"\overline{\theta}",r"\omega",'B_{0}','H'])

    """Therorical intervals [0,xmax]"""
    XLIMS = np.array([ [-0.05, 1], [0, 31],[-0.05, 1],  [-0.05, 1.2], [-0.05, 1.2]])

    DEFAULT_VALUES = np.array([0.5,15,0.5,0.5,0.5])
    """Mean default values"""

    PDF_NAMES = ["b", "theta", "omega", "", "", ""]

    DESCRIPTION = LabContextOlivine.DESCRIPTION + " Modèle de Hapke à trois paramètres, " \
                                                  "où $c$ est déduit de $b$ par $ c = \mathcal{R}(b) $"
    LABEL = LabContextOlivine.LABEL + " -$\calR$"


    def _prepare_X(self, X):
        Xfull = abstractHapkeModel._prepare_X(self, X)
        C = Crelation(Xfull[:,0])
        Xfull = np.concatenate((Xfull[:,0:1],C[:,None],Xfull[:,1:]),axis=1)
        i = np.argsort(self.HAPKE_VECT_PERMUTATION)
        return Xfull[:,i]

    def get_result(self,index_wavelength=None,full=False):
        self.partiel, tmp_partiel = None, self.partiel  # needed to avoid variable shift
        X , Std  = super().get_result(index_wavelength=index_wavelength,full=full)
        self.partiel = tmp_partiel
        return X[:,self.corrected_partiel] , Std[:,self.corrected_partiel]

    @property
    def corrected_partiel(self):
        """Returns partial indexes compatible with super class"""
        return tuple(i >= 1 and i+1 or i for i in self.partiel)


class ExperienceCRelation(SecondLearning):

    @property
    def variables_lims(self):
        c = LabContextOlivine((0,1,2,3))
        return c.variables_lims

    @property
    def variables_names(self):
        c = LabContextOlivine((0,1,2,3))
        return c.variables_names

    @property
    def variables_range(self):
        c = LabContextOlivine((0,1,2,3))
        return c.variables_range

    def clean_X(self,X,as_np_array=False):
        if self.partiel:
            var_lims =  LabContextOlivine_C.XLIMS[self.partiel,:]
        else:
            var_lims = LabContextOlivine_C.XLIMS
        mask = [len(x) > 0 and np.all((x <= var_lims[:,1]) * (var_lims[:,0] <= x)) for x in X]
        X = [x for x,b in zip(X,mask) if b]
        if as_np_array:
            X = np.array(X)
        return X , mask


    def clean_modal_prediction(self, G:GLLiM, nb_component=None, threshold=None):
        assert self.partiel[0] == 0 # or we cant retrieve C, no point in that
        X, Y , Xtest , nb_valid = super().clean_modal_prediction(G,nb_component,threshold)
        l = []
        for xs in X:
            C = Crelation(xs[:,0])
            l.append(np.concatenate((xs[:,0:1],C[:,None],xs[:,1:]),axis=1))
        return l , Y , Xtest , nb_valid

    def _one_X_prediction(self, gllim: GLLiM, Y, method):
        assert self.partiel[0] == 0  # or we cant retrieve C, no point in that
        X = super()._one_X_prediction(gllim,Y,method)
        C = Crelation(X[:,0])
        return np.concatenate((X[:,0:1],C[:,None],X[:,1:]),axis=1)



if __name__ == '__main__':
    exp = SecondLearning(LabContextOlivine_C, partiel=(0, 1, 2))
    exp.load_data(regenere_data=False,with_noise=50,N=150000,method="sobol")
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(100,mode="r",with_GMM=True,track_theta=True,init_local=None ,
                           sigma_type="full",gamma_type="full")

    X0 = exp.Xtest[1]
    Y0 = exp.context.F(X0[None, :])

    # exp.mesures.plot_retrouveY(gllim,[2,4,0.01,0.05])
    # exp.mesures.compareF(gllim)
    #
    # exp.Xtest = exp.Xtest * 0.8 + (exp.context.variables_lims[:,1] - exp.context.variables_lims[:,0]) / 10
    # exp.Ytest = exp.context.F(exp.Xtest)
    # worst_X = exp.mesures.plot_modal_prediction(gllim, [0.001])
    # worst_X = exp.mesures.plot_mean_prediction(gllim)


    # exp.mesures.plot_conditionnal_density(gllim, Y0, X0, sub_densities=4, with_modal=True, colorplot=True,
    #                                       modal_threshold=0.001)


    # MCMC_X, Std = exp.context.get_result()
    # exp.mesures.plot_density_sequence(gllim,exp.context.get_observations(), exp.context.wave_lengths,
    #                                   index=2,Xref=MCMC_X,StdRef=Std,with_pdf_images=True,varlims=(-0.2,1.2),regul="exclu")
    #


    exp2 = ExperienceCRelation(LabContextOlivine_C,partiel=(0,1,2))
    exp2.load_data(regenere_data=False,with_noise=50,N=150000,method="sobol")
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim2 = exp2.load_model(100,retrain=False,with_GMM=True,track_theta=True,init_uniform_ck=False ,
                           sigma_type="full",gamma_type="full")

    x_points = np.linspace(0,1,200)
    y = Crelation(x_points)
    exp2.mesures.correlations2D(gllim2, exp.context.get_observations(), exp.context.wave_lengths, 2, method="mean",
                                varlims=((-0.2,1),(-1,3),(0,30),(0,1)),
                                add_points={(0,1):(x_points,y)})

    # c = LabContextOlivine(partiel=(0,1,2,3))
    # exp2.mesures.plot_retrouveY(gllim2,[0.1,0.01,2,4],ref_function=c.F)

