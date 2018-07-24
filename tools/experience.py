"""Runs severals real tests on GLLiM, sGLLiM etc... """
import time
from typing import Union

import numpy as np

from Core import training
from Core.dgllim import dGLLiM
from Core.gllim import GLLiM
from experiences.rtls import RtlsCO2Context
from tools import context
from tools.archive import Archive
from tools.measures import Mesures, VisualisationMesures
from tools.results import Results, VisualisationResults

Ntest = 50000

class Experience():
    context: context.abstractFunctionModel
    archive: Archive
    mesures: VisualisationMesures
    results: VisualisationResults

    def __init__(self, context_class, partiel=None, verbose=True, with_plot=False, **kwargs):
        """If with_plot is False, methods which use matplotlib or vispy can't be used.
        Used to speed up import (no costly import)"""
        self.only_added = False
        self.adding_method = None
        self.para_learning = False
        self.partiel = partiel
        self.verbose = verbose

        self.context = context_class(partiel,**kwargs)

        self.archive = Archive(self)
        if with_plot:
            self.mesures = VisualisationMesures(self)
            self.results = VisualisationResults(self)
        else:
            self.mesures = Mesures(self)
            self.results = Results(self)

    def load_data(self, regenere_data=False, with_noise=None, N=1000, method="latin"):
        self.Nadd = 0
        self.with_noise = with_noise
        self.method = method
        self.N = N
        Ndata = N + Ntest

        if regenere_data:
            X, Y = self.context.get_data_training(Ndata,method=method)

            if with_noise:
                Y = self.context.add_noise_data(Y,std=with_noise)

            self.archive.save_data(X,Y)
        else:
            X,Y = self.archive.load_data()

        self.Xtrain, self.Ytrain = X[0:N], Y[0:N]
        self.Xtest, self.Ytest = X[-Ntest:], Y[-Ntest:]

        # Mean of training responses
        self.Xmean = self.Xtrain.mean(axis=0)

    @property
    def Ntest(self):
        return len(self.Ytest)

    @property
    def variables_lims(self):
        return self.context.variables_lims

    @property
    def variables_names(self):
        return self.context.variables_names

    @property
    def variables_range(self):
        return self.context.variables_range

    @property
    def meta_data(self):
        """Collect and returns meta data used for the current experiment"""
        return dict(
            with_noise=self.with_noise, N=self.N, Ntest=self.Ntest, K=self.K, Lw=self.Lw,
            sigma_type=self.sigma_type, gamma_type=self.gamma_type,
            gllim_class = self.gllim_cls.__name__, context=self.context.__class__.__name__,
            partiel=self.partiel , method=self.method, init_local=self.init_local,
            added=self.adding_method, Nadd=self.Nadd, para=self.para_learning
        )


    def get_infos(self,**kwargs):
        return dict(**self.meta_data, **kwargs)


    def add_data_training(self,new_data=None,adding_method='threshold',only_added=False,Nadd=None):
        self.adding_method = adding_method
        self.only_added = only_added
        if new_data:
            X,Y = new_data
            self.Nadd = X.shape[0]
            self.archive.save_data(X,Y)
        else:
            self.Nadd = Nadd
            X,Y = self.archive.load_data()

        if only_added:
            self.Xtrain = X
            self.Ytrain = Y
            self.N = 0
        print("Training data size after adding : ",self.N +self.Nadd)
        self.Xtrain = np.concatenate((self.Xtrain,X), axis=0)
        self.Ytrain = np.concatenate((self.Ytrain,Y), axis=0)
        # Mean of training responses
        self.Xmean = self.Xtrain.mean(axis=0)
        return X,Y

    def _load_gllim(self,params):
        """Create instance of gllim, load params and dimensions. Don't proceed to inversion."""
        gllim = self.gllim_cls(len(params["pi"]), self.Lw, sigma_type=self.sigma_type, gamma_type=self.gamma_type,
                               verbose=self.verbose)
        gllim.D = len(params["A"][0])
        L = len(params["Gamma"][0])
        gllim.D = len(params["A"][0])
        gllim.Lt = L - self.Lw
        gllim._init_from_dict(params)
        return gllim

    def load_model(self, K, Lw=0, sigma_type="full", gamma_type="full", gllim_cls=GLLiM,
                   mode="r", multi_init=True, init_local=None, track_theta=False, with_time=False):
        self.K = K
        self.Lw = Lw
        self.sigma_type = sigma_type
        self.gamma_type = gamma_type
        self.gllim_cls = gllim_cls
        if init_local is not None:
            multi_init = True
        self.multi_init = multi_init
        self.init_local = init_local

        if mode == "l": #load from memory
            params = self.archive.load_gllim()
            gllim = self._load_gllim(params)
            training_time = params["training_time"]
        elif mode == "r": # new training
            t = time.time()
            gllim = self.new_train(track_theta=track_theta)
            training_time = time.time() - t
            self.archive.save_gllim(gllim,track_theta,training_time=training_time)
        else: # only register meta-data
            return
        gllim.inversion()
        if with_time:
            return gllim, training_time
        return gllim

    def new_train(self,track_theta=False):
        if self.init_local:
            def ck_init_function():
                return self.context.get_X_uniform(self.K)

            gllim = training.init_local(self.Xtrain, self.Ytrain, self.K, ck_init_function, self.init_local, Lw=self.Lw,
                                        sigma_type=self.sigma_type, gamma_type=self.gamma_type,
                                        track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        elif self.multi_init:
            gllim = training.multi_init(self.Xtrain, self.Ytrain, self.K, Lw=self.Lw,
                                        sigma_type=self.sigma_type, gamma_type=self.gamma_type,
                                        track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        else:
            gllim = training.basic_fit(self.Xtrain, self.Ytrain, self.K, Lw=self.Lw,
                                       sigma_type=self.sigma_type, gamma_type=self.gamma_type,
                                       track_theta=track_theta, gllim_cls=self.gllim_cls, verbose=self.verbose)
        return gllim

    def centre_data_test(self):
        self.Xtest = self.Xtest * 0.8 + (self.context.variables_lims[:, 1] - self.context.variables_lims[:, 0]) / 10
        self.Ytest = self.context.F(self.Xtest)


    def clean_X(self,X,as_np_array=False):
        mask = self.context.is_X_valid(X)
        X = [x[m] for x,m in zip(X,mask) if (m is not None and len(x[m]) > 0) ]
        if as_np_array:
            X = np.array(X)
        return X , mask

    def clean_modal_prediction(self, G:GLLiM, nb_component=None, threshold=None):
        """Modal predicts and removes theoretically absurd prediction"""
        t = time.time()
        X , _ , _  = G.modal_prediction(self.Ytest,components=nb_component,threshold=threshold,sort_by="weight")
        if self.verbose:
            print("Gllim modal prediction done in {:.2f} secs".format(time.time() - t))
        X , mask = self.clean_X(X)
        nb_valid = sum(m.sum() if m is not None else 0 for m in mask) / sum(len(m) if m is not None else 1 for m in mask)
        mask = [m is not None for m in mask] #only X,Y for which at least one prediction is clean
        Xtest = self.Xtest[mask]
        Y = self.Ytest[mask]
        return X , Y , Xtest , nb_valid


    def _one_X_prediction(self, gllim: GLLiM, Y, method):
        if method == "mean":
            X = gllim.predict_high_low(Y)
        else:
            if method == "height":
                Xlist, _ , _  = gllim.modal_prediction(Y, components=1,sort_by="height")
            elif method == "weight":
                Xlist, _, _ = gllim.modal_prediction(Y, components=1, sort_by="weight")
            else:
                raise ValueError("Unknow prediction method")
            X = np.array([xs[0] for xs in Xlist])
        return X

    def compute_FXs(self, Xs, ref_function=None):
        ref_function = ref_function or self.context.F
        N = len(Xs)
        cumlenths = np.cumsum([len(X) for X in Xs])
        Xglue = np.array([x for X in Xs for x in X])
        Yall = ref_function(Xglue)
        Ys = []
        for i in range(N):
            debut = 0 if i == 0 else cumlenths[i - 1]
            fin = cumlenths[i]
            Ys.append(Yall[debut:fin])
        return Ys

    def best_Y_prediction(self, gllim: GLLiM, Y, ref_function=None):
        """Compute modal prediction then choose x for which F(x) is closer to y"""
        Xlist, _, _ = gllim.modal_prediction(Y, components=10, sort_by="weight")
        Ylist = np.array(self.compute_FXs(Xlist, ref_function))
        indexes = np.abs(Ylist - Y[:, :, None]).max(axis=2).argmin(axis=1)
        return Xlist[indexes]


    def reconstruct_F(self,gllim,X):
        clusters, rnk = gllim.predict_cluster(X, with_covariance=False)
        N, _ = X.shape
        # Mean estimation
        Y_estmean = np.empty((N, gllim.D))
        for n, xn in enumerate(X):
            Y_estmean[n] = np.sum(rnk[n][:, None] * (np.matmul(gllim.AkList, xn) + gllim.bkList), axis=0)

        return Y_estmean, rnk


class DoubleLearning(Experience):
    """Implements double learning methods"""

    def _predict_threshold_add(self, gllim: GLLiM, Y, threshold=0.03, nb_per_X=3):
        Xs = gllim.modal_prediction(Y,threshold=threshold)
        Xadd = (np.random.random_sample((nb_per_X, 6)) -0.5) * (self.variables_lims[:, 1] - self.variables_lims[:, 0]) / 5
        Xadd = [x2 for xsn in Xs for xs in xsn for x2 in Xadd + xs]
        Xadd, _ = self.clean_X(Xadd,as_np_array=True)
        Yadd = self.context.F(Xadd)
        return Xadd, Yadd

    def _predict_sample_add(self, gllim: GLLiM, Y, nb_per_Y=10):
        Xadd = gllim.predict_sample(Y,nb_per_Y=nb_per_Y)
        Xadd = np.array([x for Xs in Xadd for x in Xs])
        Xadd , _ = self.clean_X(Xadd,as_np_array=True)
        Yadd = self.context.F(Xadd)
        return Xadd, Yadd


    def extend_training(self, gllim: GLLiM, new_K,
                        Y=None, factor=3, threshold=None, only_added=False, track_theta=False):
        if Y is None:
            Y = self.Ytest
        if threshold:
            Xadd, Yadd = self._predict_threshold_add(gllim, Y, threshold=threshold, nb_per_X=factor)
            adding_method = "threshold_perY:{}".format(factor)
        else:
            Xadd, Yadd = self._predict_sample_add(gllim,Y,nb_per_Y=factor)
            adding_method = "sample_perY:{}".format(factor)

        if self.with_noise:
            Yadd = self.context.add_noise_data(Yadd,std = self.with_noise)

        self.add_data_training(new_data=(Xadd, Yadd), adding_method=adding_method,only_added=only_added)
        self.K = new_K
        g = self.new_train(retrain=True, track_theta=track_theta)
        return g

    def _predict_threshold_parallel(self, gllim :GLLiM, Y, threshold, nb_per_X, clusters_per_X, X=None):
        """Returns a list of tuples X,K """
        Xs = gllim.modal_prediction(Y, threshold=threshold)
        Xadd = (np.random.random_sample((nb_per_X, gllim.L)) -0.5) * (self.variables_lims[:, 1] - self.variables_lims[:, 0]) / 15
        newXYK = []
        Yclean = []
        mask = []
        for xsn , y in zip(Xs,Y):
            xadd  = np.array([x for xs in xsn for x in Xadd + xs])
            K = len(xsn) * clusters_per_X
            xadd , _ = self.clean_X(xadd,as_np_array=True)
            is_ok =len(xadd) > 0
            if is_ok:
                yadd = self.context.F(xadd)
                if self.with_noise:
                    yadd = self.context.add_noise_data(yadd,std= self.with_noise)
                newXYK.append((xadd,yadd,K))
                Yclean.append(y)
            mask.append(is_ok)
        if X is not None:
            X = X[mask]
        return newXYK,np.array(Yclean),X

    def _predict_sample_parallel(self, gllim :GLLiM, Y, nb_per_Y, K, X=None):
        Xs = gllim.predict_sample(Y,nb_per_Y=nb_per_Y)
        newXYK = []
        Yclean = []
        mask = []
        for xadd , y in zip(Xs,Y):
            xadd , _ = self.clean_X(xadd,as_np_array=True)
            is_ok = len(xadd) > 0
            if is_ok:
                yadd = self.context.F(xadd)
                if self.with_noise:
                    yadd = self.context.add_noise_data(yadd,std= self.with_noise)
                newXYK.append((xadd,yadd,K))
                Yclean.append(y)
            mask.append(is_ok)
        if X is not None:
            X = X[mask]
        return newXYK, np.array(Yclean), X


    def _get_adding_method(self, mode, nb_per_X, clusters_per_X, threshold=None):
        if mode == "s":
            return "sample_perX:{}:{}".format(nb_per_X,clusters_per_X)
        elif mode == "t":
            return "threshold{}_perX:{}:{}".format(threshold,nb_per_X,clusters_per_X)
        else:
            raise ValueError("Second data generation mode unknown")

    def extend_training_parallel(self, gllim :GLLiM, Y=None, X=None, threshold=None, nb_per_X=100, clusters_per_X=12):
        self.only_added = True
        self.para_learning = True
        if Y is None:
            Y = self.Ytest
            X = self.Xtest

        t = time.time()
        print("Modal prediction and data preparation...")
        if threshold is None:
            newXYK , Y, X= self._predict_sample_parallel(gllim,Y,nb_per_X*4,clusters_per_X*4,X=X)
            self.adding_method = self._get_adding_method("s", nb_per_X, clusters_per_X)
        else:
            newXYK ,Y , X= self._predict_threshold_parallel(gllim,Y,threshold,nb_per_X,clusters_per_X,X=X)
            self.adding_method = self._get_adding_method("t", nb_per_X, clusters_per_X, threshold=threshold)

        print("Modal prediction time {0:.2f} s".format(time.time() - t))

        gllims = training.second_training_parallel(newXYK)

        self.Nadd = len(Y)
        self.archive.save_second_learned(gllims,Y,X)

        return Y,X,gllims

    def load_second_learning(self,Nadd,threshold,nb_per_X,clusters_per_X,withX=True):
        self.Nadd = Nadd
        self.only_added = True
        self.para_learning = True
        if threshold is None:
            self.adding_method = self._get_adding_method("s", nb_per_X, clusters_per_X)
        else:
            self.adding_method = self._get_adding_method("t", nb_per_X, clusters_per_X, threshold=threshold)

        Y, X , thetas = self.archive.load_second_learned(withX)
        gllims = []
        for theta in thetas:
            gllim = self._load_gllim(theta,False)
            gllim.inversion()
            gllims.append(gllim)
        return Y, X , gllims




def monolearning():
    exp = DoubleLearning(context.LabContextOlivine, partiel=(0, 1, 2, 3))
    exp.load_data(regenere_data=False, with_noise=50, N=100001, method="sobol")
    exp.Xtrain = exp.Xtrain[:,(0,)]
    gllim = exp.load_model(100, retrain=False, with_GMM=True, track_theta=True, init_uniform_ck=False)
    exp.context.partiel = (0,)
    exp.mesures.correlations2D(gllim, exp.context.get_observations(), exp.context.wave_lengths, 1, method="mean")


def double_learning():
    exp = DoubleLearning(context.LabContextOlivine, partiel=(0, 1, 2, 3))
    exp.load_data(regenere_data=False,with_noise=50,N=10000,method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(2000,mode="l",with_GMM=False,track_theta=False,init_local=50000,
                           sigma_type="iso",gamma_type="full",gllim_cls=dGLLiM)


    # exp.extend_training_parallel(gllim,Y=exp.context.get_observations(),X=None,threshold=None,nb_per_X=5000,clusters_per_X=10)
    # Y ,X , gllims = exp.load_second_learning(64,None,5000,20,withX=False)

    index = 3
    # X0 = exp.Xtest[56]
    # Y0 = exp.context.F(X0[None, :])
    # exp.mesures.plot_conditionnal_density(gllim, Y0, None, sub_densities=4, with_modal=True, colorplot=True)

    # modals, _ , weights = gllim.modal_prediction(exp.context.get_observations(), threshold=0.0001)
    # Xw_clus = clustered_mean(modals,[w for p in weights for h,w in p])
    # Xh_clus = clustered_mean(modals,[h for p in weights for h,w in p])

    MCMC_X, Std = exp.context.get_result()




    exp.mesures.plot_density_sequence(gllim,exp.context.get_observations(), exp.context.wave_lengths,
                                      index=index,Xref=MCMC_X,StdRef=Std,with_pdf_images=True,varlims=(-0.2,1.2),regul="exclu")


    # exp.mesures.plot_density_sequence_parallel(gllims,Y, exp.context.wave_lengths,
    #                                  Xref= MCMC_X,StdRef=Std,with_pdf_images=True,index=index,varlims=(-0.2,1.2),
    #                                            regul="exclu")

    # exp.mesures.plot_density_sequence_clustered(gllim,exp.context.get_observations(),Xw_clus,Xh_clus,
    #                                             exp.context.wave_lengths, index=index,
    #                                             Xref=MCMC_X,StdRef=Std,with_pdf_images=True,varlims=(-0.2,1.2))


def test_map():
    exp = DoubleLearning(context.HapkeContext, partiel=None)
    exp.load_data(regenere_data=False, with_noise=50, N=10000, method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    gllim = exp.load_model(1000, mode="l", track_theta=False, init_local=500,
                           gllim_cls=dGLLiM)

    # Y = exp.context.get_observations()
    # latlong, mask = exp.context.get_spatial_coord()
    # Y = Y[mask] #cleaning
    # MCMC_X ,Std = exp.context.get_result(with_std=True)
    # MCMC_X = MCMC_X[mask]
    # Std = Std[mask]

    print(gllim.ckList[:, (4, 5)])
    # exp.mesures.plot_density_sequence(gllim,Y[0:10],np.arange(10),index=0,Xref=MCMC_X[0:10],StdRef=Std[0:10])

    # exp.mesures.map(gllim,Y,latlong,0,Xref = MCMC_X)

    # print(exp.context.get_result(full=True)[mask,9])

def main():
    exp = DoubleLearning(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)

    exp.load_data(regenere_data=False, with_noise=50, N=10000, method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(1000, mode="l", track_theta=False, init_local=500,
                           sigma_type="full", gamma_type="full", gllim_cls=dGLLiM)


    # exp.extend_training_parallel(gllim,Y=exp.context.get_observations(),X=None,threshold=None,nb_per_X=5000,clusters_per_X=20)
    # Y ,X , gllims = exp.load_second_learning(64,None,5000,20,withX=False)
    exp.mesures.plot_mean_prediction(gllim)
    # show_projections(exp.Xtrain)
    # exp.mesures.plot_mesures(gllim)
    # X0= exp.mesures.plot_mean_prediction(gllim)
    X0 = exp.Xtest[87]
    Y0 = exp.context.F(X0[None, :])
    # exp.mesures.plot_density_X(gllim, with_modal=False,resolution=400,colorplot=True)
    # exp.mesures.plot_conditionnal_density(gllim, Y0, X0, sub_densities=4, with_modal=True, colorplot=False)
    # exp.mesures.plot_conditionnal_density(gllim, Y0, X0, sub_densities=4, with_modal=True, colorplot=True)
    # exp.mesures.plot_conditionnal_density(gllim, Y0, X0, sub_densities=4, with_modal=True,dim=1)

    # exp.mesures.plot_density_X(gllim)
    # exp.mesures.plot_conditionnal_density(gllim, Y0, X0)
    # exp.mesures.plot_modal_prediction(gllim,[0.02])

    # print(exp.mesures.run_mesures(gllim))
    # exp.mesures.plot_estimatedF(gllim,[0,4,9]) #Dim 2 only



    # Y0 = exp.context.get_observations()[20:21,:]
    # x_MCMC = exp.context.get_result()
    # x_gllim = gllim.predict_high_low(exp.context.get_observations())
    # print(exp.mesures._relative_error(x_gllim,x_MCMC))

    varlims = [(0, 1), (-0.2, 1), (0, 25), (0, 1)]
    # exp.mesures.correlations(gllim,exp.context.get_observations(),exp.context.wave_lengths,1,method="mean",varlims=varlims)
    # exp.mesures.correlations(gllim,exp.context.get_observations(),exp.context.wave_lengths,2,method="weight",varlims=varlims)


    # MCMC_X , Std = exp.context.get_result()
    # exp.mesures.plot_density_sequence(gllim,exp.context.get_observations(),exp.context.wave_lengths,
    #                                               index=0,Xref=MCMC_X,StdRef=Std,with_pdf_images=True,
    #                                               varlims=(-0.2,1.2),regul="exclu")


    # exp.extend_training(gllim,100,Y = exp.context.get_observations(),factor=2000,only_added=True,track_theta=True)
    # gllim2 = gllim


    # gllim2 = gllims[81]
    # Y0 = Y[81][None,:]
    # X0= X[81]

    # exp.mesures.plot_modal_prediction_parallel(gllims,Y,X,[2,4,0.01,0.05])
    # exp.mesures.plot_retrouveY_parallel(gllims,Y,[2,4,0.01,0.05])




    # #
    # X = gllim2.modal_prediction(Y0,components=4)[0]
    # Y = exp.CONTEXT_CLASS.F(X,partiel=exp.partiel)
    # print(exp.mesures._relative_error(Y,Y0))
    # X = gllim2.modal_prediction(Y0,threshold=0.02)[0]
    # Y = exp.CONTEXT_CLASS.F(X,partiel=exp.partiel)
    # print(exp.mesures._relative_error(Y,Y0))


def glace():
    exp = DoubleLearning(context.VoieS, partiel=(0, 1, 2, 3))
    exp.load_data(regenere_data=False,with_noise=50,N=10000,method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(300,mode="l",track_theta=False,init_local=500,
                           sigma_type="iso",gamma_type="full",gllim_cls=dGLLiM)


    index = 0
    #
    # # MCMC_X, Std = exp.context.get_result()
    #
    # for i in [0,100,300,700,900]:
    Y , mask = exp.context.get_observations_fixed_coord(900)
    #     Y = Y[mask]
    #     # print(len(mask))
    exp.mesures.plot_density_sequence(gllim,Y, exp.context.wave_lengths,
                                      index=index,Xref=None,StdRef=None,with_pdf_images=False,varlims=(0.7,1.2),
                                      regul="exclu")
        # savepath = "VoieS_spatial{}.png".format(i)
        # exp.mesures.prediction_by_components(gllim, Y, exp.context.wave_lengths, regul="exclu",
        #                                      varlims=[(0.7,1),(0,30),(0,1),(0,1)],filename=savepath)


    # exp.mesures.plot_mesures(gllim)

    # Y , maskY = exp.context.get_observations_fixed_wl(10)
    # latlong , maskS = exp.context.get_spatial_coord()
    # mask = maskS * maskY
    # latlong = latlong[mask]
    # Y = Y[mask]
    # MCMC_X ,Std = exp.context.get_result(with_std=True)
    # MCMC_X = MCMC_X[mask]
    # Std = Std[mask]

    # exp.mesures.plot_density_sequence(gllim,Y[0:10],np.arange(10),index=0,Xref=MCMC_X[0:10],StdRef=Std[0:10])

    # exp.mesures.map(gllim,Y,latlong,0,Xref = None)


def RTLS():
    exp = Experience(RtlsCO2Context, partiel=(0, 1, 2, 3))
    exp.load_data(regenere_data=True,with_noise=50,N=10000)
    dGLLiM.dF_hook = exp.context.dF
    gllim = exp.load_model(500,mode="r",track_theta=False,init_local=500,
                           sigma_type="iso",gamma_type="full",gllim_cls=dGLLiM)

    # exp.mesures.plot_mesures(gllim)
    Xmean, Covs = exp.mesures.prediction_by_components(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                                         regul="exclu",varlims=[(0.5,1),(0,30),(0,1),(0,1)])

    exp.archive.save_resultat({"w_mean":Xmean[:,0],"w_var":Covs[:,0,0],
                      "theta_mean": Xmean[:, 1], "theta_var": Covs[:, 1, 1],
                      "b_mean": Xmean[:, 2], "b_var": Covs[:, 2, 2],
                      "c_mean": Xmean[:, 3], "c_var": Covs[:, 3, 3],
                      })



if __name__ == '__main__':
    # RTLS()
    # main()
    # monolearning()
    test_map()
    # double_learning()
    # glace()
    # test_map()