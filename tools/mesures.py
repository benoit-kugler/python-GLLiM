import time

import h5py
import numpy as np

from Core.gllim import GLLiM
from Core.log_gauss_densities import dominant_components
from tools.graphiques import compare_retrouveY, compare_Flearned, meanx_prediction, modalx_prediction, CkAnimation, \
    map_values, simple_plot, EvolutionCluster2D
from tools.graphiques import show_clusters, show_estimated_F, plot_density2D, plot_density1D, schema_1D, Evolution1D, \
    correlations2D, correlations1D, density_sequences1D
from tools.regularization import best_K, global_regularization, step_by_step, global_regularization_exclusion


class Mesures():

    LABELS_STUDY_ERROR =  ('$|| x - x_{est}||$',
              "$\max\limits_{k} \gamma_{k}^{*}(y)|| x - (A_{k}^{*} y + b_{k}^{*} )||$",
              "$\max\limits_{k} e_{k}^{(1)}$",
              "$\max\limits_{k} e_{k}^{(2)}$",
              "$\max\limits_{k} majoration(e_{k}^{(2)} )$",
              "$\max\limits_{k} \delta_{k}$",
              "Max $||\Sigma_{k}^{*} \Gamma_{k}^{-1}||$",
              "$ \max\limits_{k} \Sigma_{k}$",
              "$\max\limits_{k} \pi_{k} $",
              "$\max\limits_{k} \\frac{\Sigma_{k}}{A_{k}\Gamma_{k}}$",
              "$\max\limits_{k} \Gamma_{k}$",
              "$\max\limits_{k} \min Sp(\Gamma_{k}^{*-1})$")

    def __init__(self, experience):
        self.experience = experience

    @staticmethod
    def _relative_error(X,trueX):
        """Euclidian relative error"""
        return np.sqrt(np.square(X - trueX).sum(axis=1) / np.square(trueX).sum(axis=1)).tolist()

    @staticmethod
    def _relative_error_by_components(X,trueX):
        """If X.shape = N,L returns L,N array"""
        return np.abs( (X - trueX) / trueX ).T

    def get_title(self,title):
        """Add contexte class name"""
        return self.experience.context.__class__.__name__  + " - " + title

    def plot_clusters(self,gllim,details_clusters=False):
        exp = self.experience
        assert len(exp.partiel) == 2
        varx ,vary  = exp.variables_names
        xlim , ylim = exp.variables_lims


        show_clusters(gllim, exp.Xtrain, exp.get_infos(), superpose=True,
                      path=exp.archive.get_path("figures",filecategorie="clusters"))

        if details_clusters:
            axes_seq = show_clusters(gllim, exp.Xtrain, exp.get_infos(), varnames=(varx, vary), xlims=(xlim, ylim))
            axes_seq.fig.show()

    def plot_estimatedF(self, gllim, components, savepath=None, title=None, **kwargs):
        exp = self.experience
        assert len(exp.partiel) == 2
        varx ,vary  = exp.variables_names
        xlim , ylim = exp.variables_lims

        Yest, rnk = exp.reconstruct_F(gllim, exp.Xtrain)

        N = 100
        bh , H = exp.context.Fsample(N)
        x = bh[:,0].reshape((N,N))
        y = bh[:,1].reshape((N,N))

        data_trueF = (x , y , H)

        savepath = savepath or exp.archive.get_path("figures", filecategorie="estimatedF:weight")
        title = title or self.get_title("Estimated F - Method : mean")
        show_estimated_F(exp.Xtrain, Yest, components, data_trueF, rnk, (varx, vary), (xlim, ylim), title=title,
                         savepath=savepath, context=exp.get_infos(), **kwargs)

        # show_estimated_F(exp.Xtrain, Yh, components, data_trueF, exp.get_infos(), clusters= clusters_h,
        #                  varnames=(varx,vary), xlims = (xlim,ylim), title=self.get_title("Estimated F - Method : heights"),
        #                  savepath=exp.archive.get_path("figures",filecategorie="estimatedF:height"))
        #
        # show_estimated_F(exp.Xtrain, Ym, components, data_trueF, exp.get_infos(),
        #                  varnames=(varx,vary), xlims = (xlim,ylim), title=self.get_title("Estimated F - Method : mean" ),
        #                  savepath=exp.archive.get_path("figures",filecategorie="estimatedF:mean"))

    def plot_density_X(self, gllim : GLLiM, colorplot=True, with_modal=True,**kwargs):
        """Plot the general density of X (2D), for the given marginals (index in X)"""
        exp = self.experience
        nb_var = exp.partiel and len(exp.partiel) or 6
        fs , xlims, ylims, modal_preds,varnames, titles = [],[],[],[],[],[]
        threshold = 0.03
        #TODO : Cas d'e densité à une varaible
        # if type(marginals) is int:
        #     component = self.partiel and self.partiel[marginals] or marginals
        #     marginals = (marginals,)
        #     filename = self.filepath_figure("density-{}".format(component)) + ".png"
        #     title = "Marginal density of ${}$".format(self.CONTEXT_CLASS.X_VARIABLES_NAMES[component])
        #     x_max = self.CONTEXT_CLASS.X_VARIABLES_MAX[component]

        for i in range(nb_var):
            for j in range(i + 1, nb_var):
                componentx = exp.partiel and exp.partiel[i] or i
                componenty = exp.partiel and exp.partiel[j] or j
                varx , vary = exp.context.PARAMETERS[[componentx,componenty]]
                title = "Prior density of ${},{}$".format(varx, vary)
                xlim , ylim  = exp.context.XLIMS[[componentx , componenty]]

                def density(x_points,i=i,j=j):
                    return gllim.X_density(x_points, marginals=(i,j)), ""

                modal_pred = ()
                if with_modal:
                    h, w, modal_pred, _ = zip(*(dominant_components(gllim.pikList,
                                                                     gllim.ckList, gllim.GammakList)[0:30]))
                    modal_pred = np.array(modal_pred)[:, (i,j)]
                    modal_pred = list(zip(modal_pred,h,w))

                fs.append(density)
                xlims.append(xlim)
                ylims.append(ylim)
                modal_preds.append(modal_pred)
                varnames.append((varx,vary))
                titles.append(title)


        filename = exp.archive.get_path("figures",filecategorie="densityX-{}".format(colorplot and "color" or "contour"))
        main_title = "Marginal prior densities of X"

        plot_density2D(fs, exp.get_infos(), xlims=xlims, ylims=ylims, main_title=main_title,
                       titles=titles,modal_preds=modal_preds, colorplot=colorplot,
                       filename=filename, varnames=varnames, **kwargs)

    def _collect_infos_density(self,density_full,title,X0_obs,i,j=None,modal_pred_full=None):
        exp = self.experience
        componentx = exp.partiel and exp.partiel[i] or i
        varx = exp.context.PARAMETERS[componentx]
        xlim = exp.context.XLIMS[componentx]
        if j is not None:
            componenty = exp.partiel and exp.partiel[j] or j
            ylim = exp.context.XLIMS[componenty]
            vary = exp.context.PARAMETERS[componenty]
            title = title.format(varx, vary)
        else:
            title = title.format(varx)
            ylim ,vary = None , None

        marginals = j is not None and (i,j) or (i,)
        def density(x_points):
            return density_full(x_points,marginals)

        modal_pred = ()
        if modal_pred_full:
            modal_pred, heights, weigths = modal_pred_full
            modal_pred = modal_pred[:, marginals]
            modal_pred = list(zip(modal_pred, heights, weigths))

        trueX = X0_obs
        if X0_obs is not None:
            trueX = X0_obs[[*marginals]]
        return density,xlim,ylim,modal_pred,trueX,varx,vary,title

    def plot_conditionnal_density(self, gllim : GLLiM, Y0_obs, X0_obs, dim = 2,
                                  sub_densities=0, with_modal=False, colorplot=True,
                                  modal_threshold=0.01):
        """marginals is the index in X.
        Y0_obs is a matrix with one row, X0_obs is an array"""
        exp = self.experience
        nb_var = exp.partiel and len(exp.partiel) or len(exp.context.X_MAX)
        fs , xlims, ylims, modal_preds,trueXs,varnames, titles = [],[],[],[],[],[],[]

        def density_full(x_points,marginals):
            return gllim.forward_density(Y0_obs, x_points, marginals=marginals, sub_densities=sub_densities)

        modal_pred_full = None
        if with_modal:
            m, h , w  = gllim.modal_prediction(Y0_obs, threshold=modal_threshold)
            modal_pred_full = m[0], h[0], w[0]

        for i in range(nb_var):
            if dim == 2:
                for j in range(i + 1, nb_var):
                    density, xlim, ylim, modal_pred, trueX, varx, vary, title = self._collect_infos_density(
                        density_full,"Density of ${},{}$",X0_obs,i,j=j,modal_pred_full=modal_pred_full)

                    fs.append(density)
                    xlims.append(xlim)
                    ylims.append(ylim)
                    modal_preds.append(modal_pred)
                    trueXs.append(trueX)
                    varnames.append((varx,vary))
                    titles.append(title)
            else:
                density, xlim, ylim, modal_pred, trueX, varx, vary, title = self._collect_infos_density(
                    density_full, "Density of ${}$", X0_obs, i, j=None, modal_pred_full=modal_pred_full)

                fs.append(density)
                xlims.append(xlim)
                modal_preds.append(modal_pred)
                trueXs.append(trueX)
                varnames.append((varx,))
                titles.append(title)

        main_title = "Marginal conditional densities (Modal selection : threshold {})".format(modal_threshold)

        s = (X0_obs is None) and  "X unknown" or "".join(" ${0}$ : {1:.5f} ".format(vn, v) for vn, v in zip(exp.variables_names, X0_obs))

        filename = exp.archive.get_path("figures",filecategorie="conditionnal_density-D{}-{}".format(dim,colorplot and "color" or "contour"))
        if dim == 1:
            plot_density1D(fs, exp.get_infos(), xlims=xlims,  main_title=main_title, titles=titles,
                       modal_preds=modal_preds, trueXs=trueXs,
                       var_description=s, filename=filename, varnames=varnames)
        else:
            plot_density2D(fs, exp.get_infos(), xlims=xlims, ylims=ylims, main_title=main_title, titles=titles,
                       modal_preds=modal_preds, trueXs=trueXs, colorplot=colorplot,
                       var_description=s, filename=filename, varnames=varnames)


    def _nrmse_compare_F(self, gllim, N = 100000,clean=False):
        """if clean is given, remove theoritically absurd values and returns additional nrmse"""
        exp = self.experience
        X = exp.context.get_X_sampling(N)
        Y_vrai = exp.context.F(X)

        Y_estmean, _ = exp.reconstruct_F(gllim, X)


        error_mean = self._relative_error(Y_estmean,Y_vrai)

        if clean:
            mask_mean = exp.context.is_Y_valid(Y_estmean)
            em_clean = self._relative_error(Y_estmean[mask_mean],Y_vrai[mask_mean])
            nb_valid = mask_mean.sum() / len(mask_mean)
            return error_mean, em_clean, nb_valid
        return error_mean


    def _compute_FXs(self,Xs,ref_function):
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

    def _nrmse_retrouve_Y(self, gllim, method, ref_function=None, best=False):
        exp = self.experience
        ref_function = ref_function or exp.context.F
        if type(method) is int:
            Xs, Y, _ , nb_valid = exp.clean_modal_prediction(gllim, nb_component=method)
            label = "Components : {}".format(method)
        elif type(method) is float:
            Xs, Y, _ , nb_valid = exp.clean_modal_prediction(gllim, threshold=method)
            label = "Weight threshold : {}".format(method)
        else:
            raise TypeError("Int or float required for method")

        Ys = self._compute_FXs(Xs,ref_function)
        l = [self._relative_error(ysn, yn[None, :]) for ysn, yn in zip(Ys, Y)]
        errors = [e for subl in l for e in subl]
        if best:
            errors_best = [np.min(subl) for subl in l]
            return errors, label, nb_valid, errors_best
        return errors, label , nb_valid

    def _nrmse_mean_prediction(self, gllim,clean=False):
        exp = self.experience
        X_predicted = gllim.predict_high_low(exp.Ytest)
        retrouveY = exp.context.F(X_predicted)
        if not exp.Lw == 0:
            X_predicted = X_predicted[:, 0:-exp.Lw]
        # Normalisation
        X_test = exp.context.normalize_X(exp.Xtest)
        if clean:
            mask = exp.context.is_X_valid(X_predicted)
            Xclean = X_predicted[mask] / exp.variables_range
            nrmseclean = self._relative_error(Xclean,X_test[mask])
            nb_valid = mask.sum() / len(mask)
        X_predicted = exp.context.normalize_X(X_predicted)
        nrmse = self._relative_error(X_predicted, X_test)

        nrmse_by_components = self._relative_error_by_components(X_predicted, X_test)
        i = np.argmax(nrmse)  # worst prediction

        nrmseY = self._relative_error(retrouveY,exp.Ytest)
        if clean:
            return nrmse, nrmse_by_components, exp.Xtest[i], nrmseclean , nb_valid , nrmseY
        return nrmse, nrmse_by_components, exp.Xtest[i] , nrmseY

    def _nrmse_modal_prediction(self, gllim, method):
        exp = self.experience
        if type(method) is int:
            X, Y, Xtest , nb_valid = exp.clean_modal_prediction(gllim, nb_component=method)
            label = "Components : {}".format(method)
        elif type(method) is float:
            X, Y, Xtest , nb_valid = exp.clean_modal_prediction(gllim, threshold=method)
            label = "Weight threshold : {}".format(method)
        else:
            raise TypeError("Int or float required for method")

        bestX = np.empty(Xtest.shape)
        errs = []
        for n, (xs, xtrue) in enumerate(zip(X, Xtest)):
            diff = (xs - xtrue)
            er = np.square(diff / exp.variables_range).sum(axis=1)
            errs.append((np.min(er), n))
            i = np.argmin(er)
            bestX[n] = xs[i]
        # renormlisation
        Xtest = exp.context.normalize_X(Xtest)
        bestX  =exp.context.normalize_X(bestX)
        nrmse = self._relative_error(bestX, Xtest)
        worst_index =  np.argmax(nrmse) if len(nrmse) > 0 else 0 # worst prediction
        return nrmse, label, Xtest[worst_index] , nb_valid


    def compareF(self,gllim, N = 100000):
        exp = self.experience
        error, error_height, error_mean = self._nrmse_compare_F(gllim, N=N)

        compare_Flearned(error, error_height, error_mean, exp.get_infos(Ntest=N), cut_tail = 2,
                         savepath=exp.archive.get_path("figures",filecategorie="compareF"))

    def plot_retrouveY(self, G : GLLiM, methods, ref_function=None):
        exp = self.experience
        values = []
        labels = []
        for m in methods:
            errors, label , _  = self._nrmse_retrouve_Y(G,m,ref_function=ref_function)
            values.append(errors)
            labels.append(label)

        compare_retrouveY(values, exp.get_infos(), methods=labels, cut_tail =2,
                          savepath=exp.archive.get_path("figures",
                                                         filecategorie="retrouveY{}".format(methods)))


    def plot_mean_prediction(self, G):
        exp = self.experience
        nrmse, nrmse_by_components, worst_X , _ = self._nrmse_mean_prediction(G)

        filename = exp.archive.get_path("figures",filecategorie="meanPrediction")

        meanx_prediction(nrmse, exp.get_infos(),add_errors=nrmse_by_components,
                         add_labels=exp.variables_names,savepath=filename, cut_tail=5)
        return worst_X


    def plot_modal_prediction(self, G : GLLiM, methods):
        """Find the best prediction among nb_component given and plot NRMSE
        methods is list of int or float,
        int being interpreted as a number of components and float as a weight threshold"""
        exp = self.experience
        errors = []
        labels = []
        worst_X = None
        for m in methods:
            nrmse, label, worst_X , _ = self._nrmse_modal_prediction(G,m)
            errors.append(nrmse)
            labels.append(label)


        filename = exp.archive.get_path("figures",filecategorie="modalPrediction_meth:{}".format(methods))
        modalx_prediction(errors, labels, exp.get_infos(), savepath=filename, cut_tail=5)
        return worst_X


    def plot_mesures(self, gllim):
        """Renormalize test data to avoid border issue, and runs severals tests"""
        exp = self.experience
        methods = [1,2,0.05,0.01]

        exp.centre_data_test()

        self.plot_retrouveY(gllim,methods)
        self.compareF(gllim)
        self.plot_modal_prediction(gllim, methods)
        self.plot_mean_prediction(gllim)


    def run_mesures(self, gllim):
        """Runs severals tests. It's advised to center data before. Returns results (mean, median, std)"""
        exp = self.experience

        sumup = lambda l  = None : {"mean":np.mean(l),"median":np.median(l), "std":np.std(l)}

        if exp.context.PREFERED_MODAL_PRED == "prop":
            method =1/gllim.K
        else:
            method = exp.context.PREFERED_MODAL_PRED

        errorsF, errorsF_clean, validF = self._nrmse_compare_F(gllim, clean=True)
        errorsY , _ ,validY, errorsY_best = self._nrmse_retrouve_Y(gllim,method,best=True)
        errorsMe , _ , _ ,errorsMe_clean , validMe , meanretrouveY = self._nrmse_mean_prediction(gllim,clean=True)
        errorsMo , _ , _ , validMo = self._nrmse_modal_prediction(gllim,method)

        return dict(compareF=sumup(errorsF),meanPred=sumup(errorsMe),
                    modalPred=sumup(errorsMo),retrouveY=sumup(errorsY),
                    compareF_clean = sumup(errorsF_clean),errorsMe_clean = sumup(errorsMe_clean),
                    retrouveYbest = sumup(errorsY_best), validPreds = (validMe,validMo),
                    retrouveYmean = sumup(meanretrouveY)
                    )



    def plot_modal_prediction_parallel(self, gllims : [GLLiM], Y, Xtest, methods):
        """Same as plot_modal_prediction, but with one gllim instance per Y"""
        exp = self.experience
        errors = []
        labels = []
        for m in methods:
            print("Modal prediction...")
            bestX = np.empty(Xtest.shape)
            for n ,(g,y,xtrue) in enumerate(zip(gllims,Y,Xtest)):

                if type(m) is int:
                    xpreds = g.modal_prediction(y[None,:], components=m)[0]
                elif type(m) is float:
                    xpreds = g.modal_prediction(y[None, :], threshold=m)[0]
                else:
                    raise TypeError("Int or float required for method")
                er = np.square((xpreds - xtrue) / exp.variables_range).sum(axis=1)
                print(min(er),n)
                i = np.argmin(er)
                bestX[n] = xpreds[i]

            if type(m) is int:
                label = "Components : {}".format(m)
            elif type(m) is float:
                label = "Weight threshold : {}".format(m)
            else:
                raise TypeError("Int or float required for method")

            nrmse = self._relative_error(bestX,Xtest)
            errors.append(nrmse)
            labels.append(label)

        filename = exp.archive.get_path("figures",filecategorie="modalPredictionPara_meth:{}".format(methods))
        modalx_prediction(errors, labels, exp.get_infos(), savepath=filename, cut_tail=5)


    def plot_retrouveY_parallel(self, gllims : [GLLiM], Y, methods):
        exp = self.experience
        values = []
        labels = []
        for m in methods:
            print("Modal prediction...")
            errors = []
            for n ,(g,y) in enumerate(zip(gllims,Y)):

                if type(m) is int:
                    xpreds = g.modal_prediction(y[None,:], components=m)[0]
                elif type(m) is float:
                    xpreds = g.modal_prediction(y[None, :], threshold=m)[0]
                else:
                    raise TypeError("Int or float required for method")

                xpreds , _ = exp.clean_X(xpreds,as_np_array=True)
                if len(xpreds) == 0:
                    continue
                ysn = exp.context.F(xpreds)
                err = self._relative_error(ysn, y[None, :])
                errors.extend(err)

            if type(m) is int:
                label = "Components : {}".format(m)
            elif type(m) is float:
                label = "Weight threshold : {}".format(m)
            else:
                raise TypeError("Int or float required for method")

            values.append(errors)
            labels.append(label)

        compare_retrouveY(values, exp.get_infos(), methods=labels, cut_tail =2,
                          savepath=exp.archive.get_path("figures",
                                                         filecategorie="retrouveYPara{}".format(methods)))


    def compare_sorting(self, G : GLLiM):
        """Returns the proportion of Y such as
            - X weight == Xx
            - X weight == Xy
            - Xy == Xx """
        exp = self.experience
        X, Y, Xtest , _ = exp.clean_modal_prediction(G, exp.K //  4) # 25% best components
        N , K = X.shape[0:2]
        Xglue = X.reshape((N * K, G.L))
        Ys = np.split(exp.context.F(Xglue),N)
        issameX = np.empty((N,3))
        for n, (xs, xtrue, ys, ytest) in enumerate(zip(X, Xtest, Ys,Y)):
            i = np.argmin(np.square(xs - xtrue).sum(axis=1))
            j = np.argmin(np.square(ys - ytest).sum(axis=1))
            issameX[n] = np.array([i == 0,j == 0,i == j])
        return issameX


    def show_ck_progression(self,marginals):
        """Load theta progression and build animation"""
        exp = self.experience
        thetas, LLs = exp.archive.load_tracked_thetas()
        cks = np.array([d["c"] for d in thetas])
        varnames = exp.variables_names
        varlims = exp.variables_lims
        a = CkAnimation(cks[:, :, marginals],varnames=varnames[[*marginals]],varlims=varlims[[*marginals]])


    def illustration(self,gllim,xtrue,ytest,with_clustering=False):
        """Show summary schema in case of 1D function"""
        assert gllim.D == 1 and gllim.L == 1
        N = 10000
        xlim = self.experience.context.XLIMS[0]
        x = np.linspace(*xlim,N)
        y = self.experience.context.F(x[:,None])
        modals ,_ , weights = gllim.modal_prediction(ytest,components=10)
        X = modals[0]
        modals = list(zip(X, weights[0]))

        clusters = None
        if with_clustering:
            weights = np.array(weights[0])
            _ , clusters  = best_K(X,weights)
            print(clusters)

        fck = self.experience.context.F(gllim.ckList)


        schema_1D((x,y),gllim.ckList,gllim.ckListS,gllim.AkList,gllim.bkList,self.experience.get_infos(),
                  xlims=xlim, xtrue=xtrue[0],ytest=ytest[0,0],modal_preds=modals,clusters=clusters,
                  savepath=self.experience.archive.get_path("figures",filecategorie="schema"))


    def evolution1D(self,thetas):
        exp = self.experience
        assert exp.context.D == 1 and exp.context.L == 1
        N = 10000
        xlim = exp.context.XLIMS[0]
        x = np.linspace(*xlim,N)
        y = exp.context.F(x[:,None])
        Ys , clusterss = [],[]
        for theta in thetas:
            K = len(theta["pi"])
            gllim = exp.gllim_cls(K,Lw = exp.Lw,sigma_type=exp.sigma_type,
                                              gamma_type=exp.gamma_type,verbose=False)
            gllim.init_fit(exp.Xtrain,exp.Ytrain,theta)
            gllim.inversion()
            Y , clusters ,_ ,_ , _ = exp.reconstruct_F(gllim,exp.Xtrain)
            Ys.append(Y)
            clusterss.append(clusters)
        cks = [theta["c"] for theta in thetas]
        Evolution1D((x,y),cks,exp.Xtrain,Ys,clusterss,exp.get_infos(),xlims=xlim)

    def evolution_clusters2D(self,thetas):
        exp = self.experience
        X = exp.Xtrain
        path = "/scratch/WORK/evo_cluster.mat"
        # rnks ,Xdensity = [] , []
        # for theta in thetas:
        #     K = len(theta["pi"])
        #     gllim = exp.gllim_cls(K,Lw = exp.Lw,sigma_type=exp.sigma_type,
        #                                       gamma_type=exp.gamma_type,verbose=False)
        #     gllim.init_fit(exp.Xtrain, exp.Ytrain, theta)
        #     _, rnk = gllim.predict_cluster(X)
        #     rnks.append(rnk)  # nb de cluster a afficher
        #     Xdensity.append(gllim.X_density(X))
        # # ---- TMP
        # with h5py.File(path,"w") as f:
        #     f.create_dataset("rnks",data=rnks)
        #     f.create_dataset("Xdensity",data=Xdensity)
        with h5py.File(path) as f:
            rnks = np.array(f["rnks"])
            Xdensity = np.array(f["Xdensity"])
        if X.shape[1] == 1:
            X = np.append(X,np.array([0.5] * X.shape[0]).T[:,None],axis=1)
        EvolutionCluster2D(X,rnks,Xdensity,xlim=exp.variables_lims[0],ylim=exp.variables_lims[1])


    def error_estimation(self, gllim, X):
        ti = time.time()
        context = self.experience.context
        Y = context.F(X)
        _ , gammas = gllim._helper_forward_conditionnal_density(Y)
        N , L = X.shape
        _ ,D = Y.shape
        diff,term1,term2 = np.empty((N,gllim.K,L)) ,np.empty((N,gllim.K,L)),np.empty((N,gllim.K,L))
        maj_e2 = np.empty((N,gllim.K))

        fck = np.matmul(gllim.AkList, gllim.ckList[:, :, None])[:, :, 0] + gllim.bkList
        delta = np.linalg.norm(fck - context.F(gllim.ckList), axis=1)

        vp_gs = np.linalg.eigvalsh(gllim.GammakListS)
        alpha = 1 / vp_gs.max(axis=1)
        det_gs = vp_gs.prod(axis=1)
        nSG = gllim.norm2_SigmaSGammaInv
        for k ,pik,Aks,bks,Ak,bk,ck,Sigmaks,Gammak, Sigmak in zip(range(gllim.K),gllim.pikList,gllim.AkListS,gllim.bkListS,
                                                          gllim.AkList,gllim.bkList,gllim.ckList,gllim.SigmakListS,
                                                          gllim.GammakList,gllim.SigmakList):
            ay = Aks.dot(Y.T).T

            diff[:,k,:] = ay + bks - X

            fx = Ak.dot(X.T).T + bk

            aa = Aks.dot(fx.T).T
            mat = Sigmaks.dot(np.linalg.inv(Gammak))

            term1[:,k,:] = ay - aa
            term2[:,k,:] = mat.dot((ck  - X).T).T

            ncx = np.linalg.norm(ck - X, axis=1)

            neg = -0.5 * (context.COERCIVITE_F * ncx ** 2 + delta[k] ** 2)
            pos = 1 * delta[k]  * context.LIPSCHITZ_F * ncx
            arg_exp = alpha[k] * ( neg +  pos )

            d = np.linalg.norm( fck[k] - Y ,axis=1)
            arg_exp2 = -0.5 * alpha[k]* d**2
            ex = np.exp(arg_exp2)
            assert np.all(~np.isinf(ex))
            ex = ex * pik  / np.sqrt(det_gs[k]) / (2 * np.pi) ** (D / 2)

            maj_e2[:,k] = ex * nSG[k] * ncx


        # assert np.all(gammas <= maj_e2) #attention renormalisation

        diff = gammas[:,:,None] * diff
        term1 = gammas[:,:,None] * term1
        term2 = gammas[:,:,None] * term2
        assert np.allclose(diff, term1 + term2)  # decomposition


        ecart_sum = np.linalg.norm(np.sum(diff, axis=1),axis=1).mean()
        er_cluster = np.linalg.norm(diff, axis=2).max(axis=1).mean()
        er1 = np.linalg.norm(term1, axis=2).max(axis=1).mean()
        er2 = np.linalg.norm(term2, axis=2).max(axis=1).mean()
        maj_er2 = maj_e2.max(axis=1).mean()

        s = gllim.norm2_SigmaSGammaInv.max()
        u = np.abs(gllim.AkList[:, 0, 0] * gllim.GammakList[:, 0, 0] / gllim.full_SigmakList[:, 0, 0]).max()
        sig = gllim.SigmakList.max()
        max_pi = gllim.pikList.max()
        print("{:.2f} s for average error estimation over {} samples".format(time.time() - ti,N))
        return ecart_sum,er_cluster,er1,er2 ,maj_er2, delta.max(), s, sig, max_pi,u , gllim.GammakList.max(), alpha.min()

    def evolution_approx(self,x):
        thetas, LLs = self.experience.archive.load_tracked_thetas()
        l = []
        for theta in thetas:
            gllim = self.experience.gllim_cls(self.experience.K,Lw = self.experience.Lw,sigma_type=self.experience.sigma_type,
                                              gamma_type=self.experience.gamma_type,verbose=False)
            gllim.init_fit(self.experience.Xtrain,self.experience.Ytrain,theta)
            assert gllim.D == 1 and gllim.L == 1
            gllim.inversion()
            l.append(self.error_estimation(gllim,x[None,:]))
        labels = self.LABELS_STUDY_ERROR
        simple_plot(list(zip(*l)), labels)


    def plot_correlations2D(self, gllim : GLLiM, Y, labels_value, method="mean",
                       varlims=None, add_points=None):
        """Prediction for each Y and plot 2D with labels as color"""
        X = self.experience._one_X_prediction(gllim,Y,method)
        varlims = varlims or  self.experience.variables_lims
        varnames = self.experience.variables_names
        correlations2D(X,labels_value,self.experience.get_infos(),varnames,varlims,
                       main_title= "Corrélations - Prediction mode :  {}".format(method),add_points=add_points,
                       savepath=self.experience.archive.get_path("figures",
                                                                 filecategorie="correlations-{}".format(method)))


    def _modal_regularization(self,mode,Xweight,Xheight=None):
        reg_f = {"global": global_regularization, "step": step_by_step, "exclu": global_regularization_exclusion}[mode]
        print("Regularization {} ... ".format(mode))
        t = time.time()
        Xweight = reg_f(Xweight)
        if Xheight is not None:
            Xheight = reg_f(Xheight)
        print("Done in ", time.time() - t, "s")
        return Xweight, Xheight


    def prediction_by_components(self, gllim : GLLiM, Y, labels, varlims=None, regul=None,filename=None):
        exp = self.experience
        varlims = varlims or exp.variables_lims
        savepath = exp.archive.get_path("figures",filecategorie="synthese1D",filename=filename)
        Xweight, heights, weights = gllim.modal_prediction(Y,components=3,sort_by="weight")
        Xweight = np.array(Xweight)
        if regul:
            Xweight , _ = self._modal_regularization(regul,Xweight)
        Xmean, Covs = gllim.predict_high_low(Y, with_covariance=True)
        correlations1D(Xmean,Xweight,Covs,labels,exp.get_infos(),exp.variables_names,
                       varlims, main_title="Prédiction - Vue par composants",
                       savepath=savepath)
        return Xmean, Covs



    def plot_density_sequence(self, gllim : GLLiM, Y, labels_value, index=0, varlims=None,
                              Xref=None, StdRef=None, with_pdf_images=False, regul=None, post_processing=None):

        Xs, heights, weights = gllim.modal_prediction(Y, components=None)
        Xweight , Xheight = [] , []
        for xs, ws in zip(Xs, weights):
            Xheight.append(xs[0:3])
            l = zip(xs, ws)
            l = sorted(l, key=lambda d: d[1], reverse=True)[0:3]
            Xweight.append([x[0] for x in l])
        Xweight = np.array(Xweight)
        Xheight = np.array(Xheight)

        if regul:
            Xweight ,  Xheight = self._modal_regularization(regul,Xweight,Xheight)

        self._plot_density_sequence(gllim,Y,labels_value,Xweight,Xheight,Xref,StdRef,with_pdf_images,
                                    index,threshold=0.001,varlims=varlims,post_processing=post_processing)

    def plot_density_sequence_clustered(self,gllim,Y,Xw_clus,Xh_clus,labels_value,index=0,varlims=None,
                              Xref=None,StdRef=None,with_pdf_images=False):

        Xweight = np.array([xs[0:2,:] for xs in Xw_clus])
        Xheight = np.array([xs[0:2,:] for xs in Xh_clus])

        self._plot_density_sequence(gllim,Y,labels_value,Xweight,Xheight,Xref,StdRef,with_pdf_images,
                                    index,threshold=0.001,varlims=varlims)

    def _plot_density_sequence(self,gllim,Y,labels_x,Xweight,Xheight,Xref,StdRef,
                               with_pdf_images,index,threshold=0.01,varlims=None,post_processing=None):
        exp = self.experience
        fs, xlims, ylims, modal_preds, trueXs, varnames, titles = [], [], [], [], [], [], []

        Xmean , Covs = gllim.predict_high_low(Y,with_covariance=True)
        StdMean = np.sqrt(Covs[:,index,index])

        Xs, heights, weights = gllim.modal_prediction(Y, components=None)
        for y , xs ,hs ,ws in zip(Y,Xs,heights,weights):
            Y0_obs = y[None, :]

            def density(x_points,Y0_obs=Y0_obs):
                return gllim.forward_density(Y0_obs, x_points, marginals=(index,))

            xs = np.array([x for x,w in zip(xs,ws) if w >= threshold])
            if post_processing:
                xs = post_processing(xs)
            mpred = list(zip(xs[:,index],hs,ws))
            fs.append(density)
            modal_preds.append(mpred)
        xlim = varlims or exp.variables_lims[index]
        xvar = exp.variables_names[index]

        if with_pdf_images:
            pdf_paths = exp.context.get_images_path_densities(index)
        else:
            pdf_paths = None

        if Xref is not None:
            Xref = Xref[:,index]

        if StdRef is not None:
            StdRef = StdRef[:,index]

        if post_processing:
            Xheight = np.array([ post_processing(X) for X in Xheight])
            Xweight = np.array([ post_processing(X) for X in Xweight])
            Xmean = np.array([ post_processing(X) for X in Xmean])

        density_sequences1D(fs,modal_preds,labels_x,Xmean[:,index],Xweight[:,:,index],Xheight[:,:,index],StdMean = StdMean,
                            Yref=Xref,StdRef = StdRef ,
                            title="Densities - ${}$".format(xvar),xlim=xlim,images_paths=pdf_paths,
                            savepath=exp.archive.get_path("figures",filecategorie="sequence"))


    def plot_density_sequence_parallel(self,gllims,Y,labels_x,Xref=None,StdRef=None,
                               with_pdf_images=False,index=0,threshold=0.005,varlims=None,regul=None):
        exp = self.experience
        fs, xlims, ylims, modal_preds, trueXs, varnames, titles = [], [], [], [], [], [], []

        Xmean , StdMean = [] ,[]
        Xweight, Xheight = [], []

        for y, gllim in zip(Y,gllims):
            Y0_obs = y[None, :]
            xmean, cov = gllim.predict_high_low(Y0_obs, with_covariance=True)
            StdMean.append( np.sqrt(cov[0,index, index]))
            Xmean.append(xmean[0])

            def density(x_points,gllim=gllim,Y0_obs=Y0_obs):
                return gllim.forward_density(Y0_obs, x_points, marginals=(index,))


            Xs, heights, weights = gllim.modal_prediction(Y0_obs, components=None)
            xs , hs,  ws = Xs[0], heights[0], weights[0]
            Xheight.append(xs[0:3])
            l = zip(xs, ws)
            l = sorted(l, key=lambda d: d[1], reverse=True)[0:3]
            Xweight.append([x[0] for x in l])


            xs = np.array([x for x,w in zip(xs,ws) if w >= threshold])
            mpred = list(zip(xs[:,index],ws))
            fs.append(density)
            modal_preds.append(mpred)

        Xweight = np.array(Xweight)
        Xheight = np.array(Xheight)
        Xmean = np.array(Xmean)
        StdMean = np.array(StdMean)
        xlim = varlims or exp.variables_lims[index]
        xvar = exp.variables_names[index]

        if with_pdf_images:
            pdf_paths = exp.context.get_images_path_densities(index)
        else:
            pdf_paths = None

        if regul:
            reg_f = {"global":global_regularization,"step":step_by_step,"exclu":global_regularization_exclusion}[regul]
            print("Regularization {} ... ".format(regul))
            t = time.time()
            Xweight = reg_f(Xweight)
            Xheight = reg_f(Xheight)
            print("Done in ",time.time() - t, "s")

        density_sequences1D(fs,modal_preds,labels_x,Xmean[:,index],Xweight[:,:,index],Xheight[:,:,index],StdMean = StdMean,
                            Yref=Xref[:,index],StdRef = StdRef[:,index] ,
                            title="Second dependant learning - Densities - ${}$".format(xvar),xlim=xlim,images_paths=pdf_paths,
                            savepath=exp.archive.get_path("figures", filecategorie="dependant_learning_sequence"))



    def map(self, gllim : GLLiM, Y, latlong, index, Xref=None):
        X = gllim.predict_high_low(Y)
        x = X[:,index]
        varname = self.experience.variables_names[index]
        if Xref is not None:
            Xref = Xref[:,index]

        map_values(latlong,x,addvalues=Xref,main_title="Parameter ${}$".format(varname),
                   titles=("GLLiM","MCMC"),
                   savepath=self.experience.archive.get_path("figures", filecategorie="map-{}".format(varname)))


