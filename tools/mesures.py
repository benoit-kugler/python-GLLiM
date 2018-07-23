import os
import time

import h5py
import numpy as np

from Core.gllim import GLLiM
from Core.log_gauss_densities import dominant_components
from tools.archive import Archive
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

    def _nrmse_compare_F(self, gllim):
        """if clean is given, remove theoritically absurd values and returns additional nrmse"""
        exp = self.experience
        X = exp.context.get_X_sampling(exp.Ntest)
        Y_vrai = exp.context.F(X)

        Y_estmean, _ = exp.reconstruct_F(gllim, X)

        mask_mean = exp.context.is_Y_valid(Y_estmean)

        error_mean = self._relative_error(Y_estmean, Y_vrai)
        em_clean = self._relative_error(Y_estmean[mask_mean], Y_vrai[mask_mean])

        nb_valid = mask_mean.sum() / len(mask_mean)
        return error_mean, em_clean, nb_valid



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

    def _nrmse_mean_prediction(self, gllim):
        exp = self.experience
        X_predicted = gllim.predict_high_low(exp.Ytest)
        retrouveY = exp.context.F(X_predicted)
        mask = [np.isfinite(y).all() for y in retrouveY]
        nrmseY = self._relative_error(retrouveY[mask], exp.Ytest[mask])
        if not exp.Lw == 0:
            X_predicted = X_predicted[:, 0:-exp.Lw]

        mask = exp.context.is_X_valid(X_predicted)
        Xclean = X_predicted[mask]

        # Normalisation
        X_test = exp.context.normalize_X(exp.Xtest)
        Xclean = exp.context.normalize_X(Xclean)
        X_predicted = exp.context.normalize_X(X_predicted)

        nb_valid = mask.sum() / len(mask)

        nrmse = self._relative_error(X_predicted, X_test)
        nrmseclean = self._relative_error(Xclean, X_test[mask])

        nrmse_by_components = self._relative_error_by_components(X_predicted, X_test)
        i = np.argmax(nrmse)  # worst prediction

        return nrmse, nrmse_by_components, exp.Xtest[i], nrmseclean, nb_valid, nrmseY

    def _nrmse_modal_prediction(self, gllim, method, ref_function=None):
        exp = self.experience

        ref_function = ref_function or exp.context.F

        if type(method) is int:
            Xs, Y, Xtest, nb_valid = exp.clean_modal_prediction(gllim, nb_component=method)
            label = "Components : {}".format(method)
        elif type(method) is float:
            Xs, Y, Xtest, nb_valid = exp.clean_modal_prediction(gllim, threshold=method)
            label = "Weight threshold : {}".format(method)
        else:
            raise TypeError("Int or float required for method")

        Ys = self._compute_FXs(Xs, ref_function)
        l = [self._relative_error(ysn, yn[None, :]) for ysn, yn in zip(Ys, Y)]
        errorsY = [e for subl in l for e in subl]
        errorsY_best = [np.min(subl) for subl in l]

        bestX = np.empty(Xtest.shape)
        for n, (xs, xtrue) in enumerate(zip(Xs, Xtest)):
            diff = (xs - xtrue)
            er = np.square(diff / exp.variables_range).sum(axis=1)
            i = np.argmin(er)
            bestX[n] = xs[i]
        # renormlisation
        Xtest = exp.context.normalize_X(Xtest)
        bestX  =exp.context.normalize_X(bestX)
        nrmse = self._relative_error(bestX, Xtest)
        worst_index =  np.argmax(nrmse) if len(nrmse) > 0 else 0 # worst prediction
        return nrmse, label, Xtest[worst_index], nb_valid, errorsY, errorsY_best

    def run_mesures(self, gllim):
        """Runs severals tests. It's advised to center data before. Returns results (mean, median, std)"""
        exp = self.experience

        sumup = lambda l  = None : {"mean":np.mean(l),"median":np.median(l), "std":np.std(l)}

        if exp.context.PREFERED_MODAL_PRED == "prop":
            method =1/gllim.K
        else:
            method = exp.context.PREFERED_MODAL_PRED

        errorsF, errorsF_clean, validF = self._nrmse_compare_F(gllim)
        errorsMe, _, _, errorsMe_clean, validMe, meanretrouveY = self._nrmse_mean_prediction(gllim)
        errorsMo, _, _, validMo, errorsY, errorsY_best = self._nrmse_modal_prediction(gllim, method)

        return dict(compareF=sumup(errorsF),meanPred=sumup(errorsMe),
                    modalPred=sumup(errorsMo),retrouveY=sumup(errorsY),
                    compareF_clean = sumup(errorsF_clean),errorsMe_clean = sumup(errorsMe_clean),
                    retrouveYbest = sumup(errorsY_best), validPreds = (validMe,validMo),
                    retrouveYmean = sumup(meanretrouveY)
                    )

    def compare_sorting(self, G: GLLiM):
        """Returns the proportion of Y such as
            - X best weight == X initial
            - X best weight == X best retrouve Y
            - X best retrouve Y == X initial """
        exp = self.experience
        Xs, Y, Xtest, _ = exp.clean_modal_prediction(G, exp.K // 4)  # 25% best components
        N = len(Xs)
        Ys = self._compute_FXs(Xs)
        issameX = np.empty((N, 3))
        for n, (xs, xtrue, ys, ytest) in enumerate(zip(Xs, Xtest, Ys, Y)):
            i = np.argmin(np.square(xs - xtrue).sum(axis=1))
            j = np.argmin(np.square(ys - ytest).sum(axis=1))
            issameX[n] = np.array([i == 0, j == 0, i == j])
        return issameX, issameX.mean(axis=0)


class MesuresSecondLearning(Mesures):
    # TODO : Mettre à jour
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


class VisualisationMesures(Mesures):

    def __init__(self, experience):
        super().__init__(experience)
        import plotting.graphiques as G
        self.G = G

    def get_title(self, title):
        """Add contexte class name"""
        return self.experience.context.__class__.__name__ + " - " + title

    def plot_clusters(self, gllim, details_clusters=False, indexes=(0, 1)):
        """If details_clusters plots a sequence of K clusters. Else shows the superposition.
        Uses indexes of X"""
        exp = self.experience
        varnames = exp.variables_names[indexes]
        varlims = exp.variables_lims[indexes]
        _, rnk = gllim.predict_cluster(exp.Xtrain)
        X = exp.Xtrain[indexes]

        if details_clusters:
            self.G.clusters_one_by_one(X, rnk, gllim.ckList, varnames, varlims, context=exp.get_infos(),
                                       draw_context=True,
                                       savepath=exp.archive.get_path("figures", filecategorie="clusters"))
        else:
            self.G.clusters(X, rnk, gllim.ckList, varnames, varlims, context=exp.get_infos(), draw_context=True,
                            savepath=exp.archive.get_path("figures", filecategorie="clusters"))

    def plot_estimatedF(self, gllim, components, savepath=None, title=None, **kwargs):
        exp = self.experience
        assert len(exp.partiel) == 2

        Yest, rnk = exp.reconstruct_F(gllim, exp.Xtrain)

        N = 100
        bh, H = exp.context.Fsample(N)
        x = bh[:, 0].reshape((N, N))
        y = bh[:, 1].reshape((N, N))

        data_trueF = (x, y, H)

        savepath = savepath or exp.archive.get_path("figures", filecategorie="estimatedF:weight")
        title = title or self.get_title("Estimated F - Method : mean")
        self.G.estimated_F(exp.Xtrain, Yest, components, data_trueF, rnk, exp.variables_names, exp.variables_lims,
                           title=title,
                           savepath=savepath, context=exp.get_infos(), **kwargs)

    def _collect_plot_density2D(self, density_full, base_title, trueX, modal_pred_full, filename, **kwargs):
        exp = self.experience
        nb_var = exp.partiel and len(exp.partiel) or len(exp.context.X_MAX)
        graph_datas = []

        for i in range(nb_var):
            for j in range(i + 1, nb_var):
                density, xlim, ylim, modal_pred, trueXij, varx, vary, title = \
                    self._collect_infos_density(density_full, base_title, trueX, i, j=j,
                                                modal_pred_full=modal_pred_full)
                graph_datas.append((density, xlim, ylim, modal_pred, trueXij, (varx, vary), title))
        fs, xlims, ylims, modal_preds, trueXs, varnames, titles = zip(*graph_datas)

        varlims = zip(xlims, ylims)
        colorplot = kwargs.pop("colorplot", False)
        var_description = kwargs.pop("var_description", "")
        title = kwargs.pop("main_title", "")
        self.G.plot_density2D(fs, varlims, varnames, titles, modal_preds, trueXs, colorplot,
                              var_description, context=exp.get_infos(), title=title, savepath=filename, **kwargs)

    def _collect_plot_density1D(self, density_full, base_title, trueX, modal_pred_full, filename, **kwargs):
        exp = self.experience
        nb_var = exp.partiel and len(exp.partiel) or len(exp.context.X_MAX)
        graph_datas = []

        for i in range(nb_var):
            density, xlim, ylim, modal_pred, trueXi, varx, vary, title = \
                self._collect_infos_density(density_full, base_title, trueX, i, j=None, modal_pred_full=modal_pred_full)
            graph_datas.append((density, xlim, ylim, modal_pred, trueXi, (varx, vary), title))
        fs, xlims, ylims, modal_preds, trueXs, varnames, titles = zip(*graph_datas)

        self.G.plot_density1D(fs, exp.get_infos(), xlims=xlims,
                              titles=titles, modal_preds=modal_preds, trueXs=trueXs,
                              filename=filename, varnames=varnames, **kwargs)

    def plot_density_X(self, gllim: GLLiM, colorplot=True, with_modal=True):
        """Plot the general density of X (2D), for the given marginals (index in X)"""
        exp = self.experience

        def density_full(x_points, marginals):
            return gllim.X_density(x_points, marginals=marginals), None

        base_title = "Prior density of ${},{}$"
        modal_pred_full = None
        if with_modal:
            h, w, c, _ = zip(*(dominant_components(gllim.pikList,
                                                   gllim.ckList, gllim.GammakList)[0:30]))
            modal_pred_full = np.array(c), np.array(w), np.array(h)

        filename = exp.archive.get_path("figures",
                                        filecategorie="densityX-{}".format(colorplot and "color" or "contour"))
        main_title = "Marginal prior densities of X"

        self._collect_plot_density2D(density_full, base_title, None, modal_pred_full, filename,
                                     main_title=main_title, colorplot=colorplot)

    def plot_conditionnal_density(self, gllim: GLLiM, Y0_obs, X0_obs, dim=2,
                                  sub_densities=0, with_modal=False, colorplot=True,
                                  modal_threshold=0.01):
        """marginals is the index in X.
        Y0_obs is a matrix with one row, X0_obs is an array"""
        exp = self.experience

        def density_full(x_points, marginals):
            return gllim.forward_density(Y0_obs, x_points, marginals=marginals, sub_densities=sub_densities)

        modal_pred_full = None
        if with_modal:
            m, h, w = gllim.modal_prediction(Y0_obs, threshold=modal_threshold)
            modal_pred_full = m[0], h[0], w[0]

        filename = exp.archive.get_path("figures", filecategorie="conditionnal_density-D{}-{}".format(dim,
                                                                                                      colorplot and "color" or "contour"))

        main_title = "Marginal conditional densities (Modal selection : threshold {})".format(modal_threshold)

        s = (X0_obs is None) and "X unknown" or "".join(
            " ${0}$ : {1:.5f} ".format(vn, v) for vn, v in zip(exp.variables_names, X0_obs))

        if dim == 2:
            base_title = "Density of ${},{}$"
            self._collect_plot_density2D(density_full, base_title, X0_obs, modal_pred_full, filename,
                                         main_title=main_title, colorplot=colorplot, var_description=s,
                                         draw_context=True)

        elif dim == 1:
            base_title = "Density of ${}$"
            self._collect_plot_density1D(density_full, base_title, X0_obs, modal_pred_full, filename,
                                         main_title=main_title, var_description=s)
        else:
            raise ValueError("Unknown dimension. Must be one 1 or 2")

    def plot_compareF(self, gllim):
        exp = self.experience
        nrmse, nrmse_clean, nb_valid = self._nrmse_compare_F(gllim)

        self.G.hist_Flearned([nrmse], 2, None, context=exp.get_infos(), draw_context=True,
                             savepath=exp.archive.get_path("figures", filecategorie="compareF"))

    def plot_mean_prediction(self, G):
        exp = self.experience
        nrmse, nrmse_by_components, _, nrmseclean, nb_valid, nrmseY = self._nrmse_mean_prediction(G)

        errors = np.array([nrmse] + nrmse_by_components)
        labels = ["Vector error"] + list(exp.variables_names)

        self.G.hist_meanPrediction(errors, 5, labels, context=exp.get_infos(), draw_context=True,
                                   savepath=exp.archive.get_path("figures", filecategorie="meanPrediction"))

        self.G.hist_retrouveYmean([nrmseY], 5, None, context=exp.get_infos(), draw_context=True,
                                  savepath=exp.archive.get_path("figures", filecategorie="retrouveYmean"))

    def plot_modal_prediction(self, G: GLLiM, methods, ref_function=None):
        """Find the best prediction among nb_component given and plot NRMSE for x prediction and y compatibility
        methods is list of int or float,
        int being interpreted as a number of components and float as a weight threshold"""
        exp = self.experience
        values_X = []
        values_Y = []
        values_Yb = []
        labels = []
        for m in methods:
            errorsMo, label, _, _, errorsY, errorsY_best = self._nrmse_modal_prediction(G, m, ref_function=ref_function)
            values_X.append(errorsMo)
            values_Y.append(errorsY)
            values_Yb.append(errorsY_best)
            labels.append(label)

        self.G.hist_modalPrediction(values_X, 2, labels, context=exp.get_infos(), draw_context=True,
                                    savepath=exp.archive.get_path("figures",
                                                                  filecategorie="modalPrediction_meth:{}".format(
                                                                      methods)))

        self.G.hist_retrouveY(values_Y, 2, labels, context=exp.get_infos(), draw_context=True,
                              savepath=exp.archive.get_path("figures",
                                                            filecategorie="retrouveY{}".format(methods)))

        self.G.hist_retrouveYbest(values_Yb, 2, labels, context=exp.get_infos(), draw_context=True,
                                  savepath=exp.archive.get_path("figures",
                                                                filecategorie="retrouveYbest{}".format(methods)))

    def plot_mesures(self, gllim):
        """Renormalize test data to avoid border issue, and runs severals tests"""
        exp = self.experience
        methods = [1, 2, 0.05, 0.01]

        exp.centre_data_test()

        self.plot_compareF(gllim)
        self.plot_mean_prediction(gllim)
        self.plot_modal_prediction(gllim, methods)

    def illustration(self, gllim, xtrue, ytest, savepath=None):
        """Show summary schema in case of 1D function"""
        assert gllim.D == 1 and gllim.L == 1
        N = 10000
        xlim = self.experience.context.XLIMS[0]
        x = np.linspace(*xlim, N)
        y = self.experience.context.F(x[:, None])
        modals, _, weights = gllim.modal_prediction(ytest, components=10, sort_by="weight")
        X = modals[0]
        modals = list(zip(X, weights[0]))
        savepath = savepath or self.experience.archive.get_path("figures", filecategorie="schema")
        context = dict(**self.experience.get_infos(), max_Gamma=gllim.GammakList.max())
        self.G.schema_1D((x, y), gllim.ckList, gllim.ckListS, gllim.AkList, gllim.bkList, xlim, xtrue[0], ytest[0, 0],
                         modals,
                         context=context, write_context=True, draw_context=False,
                         title="", savepath=savepath)

    def evolution_illustration(self, thetas, cached=False):
        """Show 1D summary of evolution during fitting
                If cached is True, load values from archive.BASE_PATH/evo_1D.mat """
        exp = self.experience
        assert exp.context.D == 1 and exp.context.L == 1
        N = 10000
        xlims = exp.context.XLIMS[0]
        x = np.linspace(*xlims, N)
        y = exp.context.F(x[:, None])
        points_F = (x, y)
        if not cached:
            datas = []
            for theta in thetas:
                K = len(theta["pi"])
                gllim = exp.gllim_cls(K, Lw=exp.Lw, sigma_type=exp.sigma_type,
                                      gamma_type=exp.gamma_type, verbose=False)
                gllim.init_fit(exp.Xtrain, exp.Ytrain, theta)
                gllim.inversion()
                datas.append((gllim.ckList, gllim.ckListS, gllim.AkList, gllim.bkList))
            cks, ckSs, Aks, bks = zip(*datas)
            Archive.save_evolution_1D(cks, ckSs, Aks, bks)
        else:
            cks, ckSs, Aks, bks = Archive.load_evolution_1D()
        self.G.Evolution1D(points_F, cks, ckSs, Aks, bks, xlims)

    def evolution_clusters2D(self, thetas, cached=False):
        """Load theta progression and build animation of clusters and X density evolution.
        If cached is True, load values from archive.BASE_PATH/evo_cluster.mat """
        exp = self.experience
        X = exp.Xtrain
        if not cached:
            rnks, Xdensitys = [], []
            for theta in thetas:
                K = len(theta["pi"])
                gllim = exp.gllim_cls(K, Lw=exp.Lw, sigma_type=exp.sigma_type,
                                      gamma_type=exp.gamma_type, verbose=False)
                gllim.init_fit(exp.Xtrain, exp.Ytrain, theta)
                _, rnk = gllim.predict_cluster(X)
                rnks.append(rnk)  # nb de cluster a afficher
                Xdensitys.append(gllim.X_density(X))
            Archive.save_evolution_clusters(rnks, Xdensitys)
        else:
            rnks, Xdensitys = Archive.load_evolution_clusters()
        if X.shape[1] == 1:
            X = np.append(X, np.array([0.5] * X.shape[0]).T[:, None], axis=1)
        self.G.EvolutionCluster2D(X, rnks, Xdensitys, xlim=exp.variables_lims[0], ylim=exp.variables_lims[1])
