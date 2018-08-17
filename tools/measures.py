import logging
import time

import numpy as np

from Core.gllim import GLLiM
from Core.log_gauss_densities import dominant_components
from tools.archive import Archive
import tools.experience

class Mesures():
    experience: 'tools.experience.Experience'

    LABELS_STUDY_ERROR =  ('$|| x - x_{est}||$',
                           r"$\sum\limits_{k} \frac{ \pi_{k} h_{k}}{ \sqrt{(2 \pi)^{D} \det{\Gamma_{k}^{*}}} } $",
                           "$\sum\limits_{k} \gamma_{k}^{*}(y) || A_{k}^{*}(y - c_{k}^{*}) ||$",
                           "$\sum\limits_{k} \gamma_{k}^{*}(y) || x - c_{k} || $",
              "$\max\limits_{k} \delta_{k}$",
              "Max $||\Sigma_{k}^{*} \Gamma_{k}^{-1}||$",
              "$ \max\limits_{k} \Sigma_{k}$",
              "$\max\limits_{k} \pi_{k} $",
                           r"$\max\limits_{k} \frac{A_{k}\Gamma_{k}}{\Sigma_{k}}$",
              "$\max\limits_{k} \Gamma_{k}$",
              "$\max\limits_{k} \min Sp(\Gamma_{k}^{*-1})$")

    def __init__(self, experience):
        self.experience = experience

    # @staticmethod
    # def _relative_error(X,trueX):
    #     """Euclidian relative error"""
    #     return np.sqrt(np.square(X - trueX).sum(axis=1) / np.square(trueX).sum(axis=1)).tolist()

    @staticmethod
    def _relative_error(X, trueX, with_components=False):
        """Absolute difference. Data should be normalized beforehand"""
        diff = np.abs(X - trueX)
        norm = diff.max(axis=1)
        if with_components:
            return norm, diff.T
        return norm

    @staticmethod
    def sumup_errors(l):
        """Return a dict containing mean, median and std of values"""
        return {"mean": np.mean(l), "median": np.median(l), "std": np.std(l)}


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
        X = exp.Xtest
        Y_vrai = exp.context.F(X)

        Y_estmean, _ = exp.reconstruct_F(gllim, X)

        mask_mean = exp.context.is_Y_valid(Y_estmean)
        # normalisation
        Y_estmean = exp.context.normalize_Y(Y_estmean)
        Y_vrai = exp.context.normalize_Y(Y_vrai)

        error_mean = self._relative_error(Y_estmean, Y_vrai)
        em_clean = self._relative_error(Y_estmean[mask_mean], Y_vrai[mask_mean])

        nb_valid = mask_mean.sum() / len(mask_mean)
        return error_mean, em_clean, nb_valid

    def _normalize_nrmse(self, Xpredicted, Xtrue, normalization):
        Xtrue = normalization(Xtrue)
        Xpredicted = normalization(Xpredicted)
        nrmse, nrmse_components = self._relative_error(Xpredicted, Xtrue, with_components=True)
        worst_index = np.argmax(nrmse) if len(nrmse) > 0 else 0  # worst prediction
        return nrmse, nrmse_components, worst_index

    def _nrmse_oneXperY(self, Xpredicted, Xtrue, Y, ref_function):
        exp = self.experience
        retrouveY = ref_function(Xpredicted)

        mask = [np.isfinite(y).all() for y in retrouveY]
        nrmseY, _, _ = self._normalize_nrmse(retrouveY[mask], Y[mask], exp.context.normalize_Y)

        if not exp.Lw == 0:
            Xpredicted = Xpredicted[:, 0:-exp.Lw]

        mask = exp.context.is_X_valid(Xpredicted)

        nb_valid = mask.sum() / len(mask)

        nrmse, nrmse_by_components, i = self._normalize_nrmse(Xpredicted, Xtrue, exp.context.normalize_X)
        # nrmseclean, _, _ = self._normalize_nrmse(Xpredicted[mask], Xtrue[mask], exp.context.normalize_X)

        return nrmse, nrmse_by_components, Xtrue[i], nb_valid, nrmseY

    def _nrmse_multiXperY(self, Xspredicted, Xtrue, Y, ref_function):
        exp = self.experience
        Ys = self.experience.compute_FXs(Xspredicted, ref_function)
        norm = exp.context.normalize_Y
        l = [self._relative_error(norm(ysn), norm(yn[None, :])) for ysn, yn in zip(Ys, Y)]
        nrmseY = [e for subl in l for e in subl]
        nrmseY_best = [np.min(subl) for subl in l]

        bestX = np.empty(Xtrue.shape)
        for n, (xs, xtrue) in enumerate(zip(Xspredicted, Xtrue)):
            diff = (xs - xtrue)
            i = np.square(diff / exp.variables_range).sum(axis=1).argmin()
            bestX[n] = xs[i]

        nrmse, nrmse_by_components, worst_index = self._normalize_nrmse(bestX, Xtrue, exp.context.normalize_X)

        return nrmse, nrmse_by_components, Xtrue[worst_index], nrmseY, nrmseY_best

    def _nrmse_mean_prediction(self, gllim):
        exp = self.experience
        X_predicted = gllim.predict_high_low(exp.Ytest)

        return self._nrmse_oneXperY(X_predicted, exp.Xtest, exp.Ytest, exp.context.F)

    def _nrmse_modal_prediction(self, gllim, method, ref_function=None):
        exp = self.experience

        if type(method) is int:
            Xs, Y, Xtest, nb_valid = exp.clean_modal_prediction(gllim, nb_component=method)
            label = "Components : {}".format(method)
        elif type(method) is float:
            Xs, Y, Xtest, nb_valid = exp.clean_modal_prediction(gllim, threshold=method)
            label = "Weight threshold : {}".format(method)
        else:
            raise TypeError("Int or float required for method")

        nrmse, nrmse_by_components, worstX, nrmseY, nrmseY_best = self._nrmse_multiXperY(Xs, Xtest, Y, ref_function)

        return nrmse, label, worstX, nb_valid, nrmseY, nrmseY_best

    def _nrmse_clustered_prediction(self, gllim):
        exp = self.experience
        logging.info("Starting clustered prediction...")
        Xs = gllim.clustered_prediction(exp.Ytest, exp.context.F)
        Xs, mask = exp.clean_X(Xs)
        mask = [(m is not None and sum(m, 0) > 0) for m in mask]  # only X,Y for which at least one prediction is clean
        Y, Xtrue = exp.Ytest[mask], exp.Xtest[mask]
        nrmse, _, _, nrmseY, nrmseY_best = exp.mesures._nrmse_multiXperY(Xs, Xtrue, Y, None)
        return nrmse, nrmseY, nrmseY_best

    def run_mesures(self, gllim):
        """Runs severals tests. It's advised to center data before. Returns results (mean, median, std)"""
        exp = self.experience

        if exp.context.PREFERED_MODAL_PRED == "prop":
            method = 1 / gllim.K
        else:
            method = exp.context.PREFERED_MODAL_PRED
        errorsF, errorsF_clean, validF = self._nrmse_compare_F(gllim)
        errorsMe, errorsMecomponents, _, validMe, meanretrouveY = self._nrmse_mean_prediction(gllim)
        errorsMo, labelmodal, _, validMo, errorsY, errorsY_best = self._nrmse_modal_prediction(gllim, method)
        errorsMecomponents = [self.sumup_errors(x) for x in errorsMecomponents]

        logging.debug(f'\tModal prediction mode for {exp.context.__class__.__name__} : {labelmodal}')
        return dict(compareF=self.sumup_errors(errorsF), meanPred=self.sumup_errors(errorsMe),
                    modalPred=self.sumup_errors(errorsMo), retrouveY=self.sumup_errors(errorsY),
                    compareF_clean=self.sumup_errors(errorsF_clean), retrouveYbest=self.sumup_errors(errorsY_best),
                    validPreds=(validMe, validMo), retrouveYmean=self.sumup_errors(meanretrouveY),
                    errorsMecomponents=errorsMecomponents)

    def compare_sorting(self, G: GLLiM):
        """Returns the proportion of Y such as
            - X best weight == X initial
            - X best weight == X best retrouve Y
            - X best retrouve Y == X initial """
        exp = self.experience
        Xs, Y, Xtest, _ = exp.clean_modal_prediction(G, exp.K // 4)  # 25% best components
        N = len(Xs)
        Ys = self.experience.compute_FXs(Xs)
        issameX = np.empty((N, 3))
        for n, (xs, xtrue, ys, ytest) in enumerate(zip(Xs, Xtest, Ys, Y)):
            i = np.argmin(np.square(xs - xtrue).sum(axis=1))
            j = np.argmin(np.square(ys - ytest).sum(axis=1))
            issameX[n] = np.array([i == 0, j == 0, i == j])
        return issameX, issameX.mean(axis=0)

    def error_estimation(self, gllim, X):
        ti = time.time()
        context = self.experience.context
        Y = context.F(X)
        modals, gammas, normalisation = gllim._helper_forward_conditionnal_density(Y)
        N, L = X.shape
        _, D = Y.shape
        diff, term2 = np.empty((N, gllim.K, L)), np.empty((N, gllim.K, L))


        fck = np.matmul(gllim.AkList, gllim.ckList[:, :, None])[:, :, 0] + gllim.bkList
        d = fck - context.F(gllim.ckList)
        delta = np.linalg.norm(d, axis=1)

        vp_gs = np.linalg.eigvalsh(gllim.GammakListS)
        alpha = 1 / vp_gs.max(axis=1)
        det_gs = vp_gs.prod(axis=1)

        M, Mp = context.COERCIVITE_F, context.LIPSCHITZ_F
        tk = (delta * Mp + np.sqrt(delta ** 2 * (Mp ** 2) + (M / alpha))) / M
        hk = np.exp(- 0.5 + delta * alpha * Mp * tk) * tk
        e1k = gllim.pikList * hk / np.sqrt((2 * np.pi) ** D * det_gs)

        dGInvd = -0.5 * (delta.T).dot(np.matmul(
            np.linalg.inv(gllim.GammakListS), d[:, :, None])[:, :, 0])
        e1k *= np.exp(dGInvd)

        xck, maj_x = np.empty((N, gllim.K, L)), np.empty((N, gllim.K))
        for k, Aks, ck, cks, alphak, deltak in zip(range(gllim.K), gllim.AkListS, gllim.ckList, gllim.ckListS, alpha,
                                                   delta):
            ay = Aks.dot(Y.T - cks).T
            term2[:, k, :] = ay
            xck[:, k, :] = X - ck
            ntx = np.linalg.norm(X - ck, axis=1)
            maj_x[:, k] = np.exp(-0.5 * alphak * M * ntx ** 2 + 2 * deltak * alphak * ntx * Mp) * ntx

        mean_pred = np.sum(gammas[:, :, None] * modals.transpose(1, 2, 0), axis=1)
        term2 = gammas[:, :, None] * term2

        ecart_sum = np.linalg.norm(mean_pred - X, axis=1).mean()
        er1 = (e1k.sum(axis=0) / normalisation).mean()
        er2 = np.linalg.norm(term2, axis=2).sum(axis=1).mean()
        er3 = ((gllim.pikList * maj_x / np.sqrt((2 * np.pi) ** D * det_gs) * np.exp(dGInvd)).sum(
            axis=1) / normalisation).mean()
        print("Distance au maximum de h : ", (np.linalg.norm(xck, axis=2) - tk).mean())

        sigSGInv = gllim.norm2_SigmaSGammaInv.max()
        AGSInv = np.abs(gllim.AkList[:, 0, 0] * gllim.GammakList[:, 0, 0] / gllim.full_SigmakList[:, 0, 0]).max()
        sig = gllim.SigmakList.max()
        max_pi = gllim.pikList.max()
        logging.debug("{:.2f} s for average error estimation over {} samples".format(time.time() - ti, N))
        return ecart_sum, er1, er2, er3, delta.max(), sigSGInv, sig, max_pi, AGSInv, gllim.GammakList.max(), alpha.min()


class MesuresSecondLearning(Mesures):
    experience: 'tools.experience.SecondLearning'

    def _nrmse_mean_prediction(self, gllims: [GLLiM], Y, Xtest):
        """Mean prediction errors for each gllims and Y"""
        Xpredicted = np.array([gllim.predict_high_low(y[None, :])[0] for gllim, y in zip(gllims, Y)])
        return self._nrmse_oneXperY(Xpredicted, Xtest, Y, self.experience.context.F)

    def _nrmse_modal_prediction(self, gllims: [GLLiM], Y, Xtest, method, ref_function=None):
        if type(method) is int:
            label = "Components : {}".format(method)
            Xspredicted, Y, Xtest, nb_valid = self.experience.clean_modal_prediction(gllims, Y, Xtest,
                                                                                     nb_component=method)
        elif type(method) is float:
            label = "Weight threshold : {}".format(method)
            Xspredicted, Y, Xtest, nb_valid = self.experience.clean_modal_prediction(gllims, Y, Xtest,
                                                                                     threshold=method)

        else:
            raise TypeError("Int or float required for method")

        nrmse, nrmse_by_components, worstX, nrmseY, nrmseY_best = \
            self._nrmse_multiXperY(Xspredicted, Xtest, Y, ref_function)

        return nrmse, label, worstX, nb_valid, nrmseY, nrmseY_best

    def run_mesures(self, gllims: [GLLiM], Y, Xtest):
        """Runs severals tests. It's advised to center data before. Returns results (mean, median, std)"""
        exp = self.experience

        def sumup(l):
            return {"mean": np.mean(l), "median": np.median(l), "std": np.std(l)}

        if exp.context.PREFERED_MODAL_PRED == "prop":
            method = 1 / gllims[0].K
        else:
            method = exp.context.PREFERED_MODAL_PRED

        errorsMe, _, _, validMe, meanretrouveY = self._nrmse_mean_prediction(gllims, Y, Xtest)
        errorsMo, _, _, validMo, errorsY, errorsY_best = self._nrmse_modal_prediction(gllims, Y, Xtest, method)

        return dict(compareF=None, meanPred=sumup(errorsMe),
                    modalPred=sumup(errorsMo), retrouveY=sumup(errorsY),
                    compareF_clean=None, retrouveYbest=sumup(errorsY_best),
                    validPreds=(validMe, validMo), retrouveYmean=sumup(meanretrouveY)
                    )






class VisualisationMesures(Mesures):

    def __init__(self, experience):
        super().__init__(experience)
        import plotting.graphiques as G
        self.G = G

    def get_title(self, title):
        """Add contexte class name"""
        return self.experience.context.__class__.__name__ + " - " + title

    def plot_clusters(self, gllim, details_clusters=False, indexes=(0, 1), savepath=None, **kwargs):
        """If details_clusters plots a sequence of K clusters. Else shows the superposition.
        Uses indexes of X"""
        exp = self.experience
        varnames = exp.variables_names[list(indexes)]
        varlims = exp.variables_lims[list(indexes)]
        _, rnk = gllim.predict_cluster(exp.Xtrain)
        X = exp.Xtrain[:, indexes]
        savepath = savepath or exp.archive.get_path("figures", filecategorie="clusters")

        if details_clusters:
            self.G.clusters_one_by_one(X, rnk, gllim.ckList, varnames, varlims, context=exp.get_infos(),
                                       draw_context=True,
                                       savepath=savepath)
        else:
            self.G.clusters(X, rnk, gllim.ckList, varnames, varlims, context=exp.get_infos(),
                            savepath=savepath, **kwargs)


    def plot_estimatedF(self, gllim, components, savepath=None, title=None, **kwargs):
        exp = self.experience
        assert len(exp.variables_lims) == 2

        Yest, rnk = exp.reconstruct_F(gllim, exp.Xtrain)

        N = 100
        bh, H = exp.context.Fsample(N)
        x = bh[:, 0].reshape((N, N))
        y = bh[:, 1].reshape((N, N))

        data_trueF = (x, y, H)

        savepath = savepath or exp.archive.get_path("figures", filecategorie="estimatedF:mean")
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

        errors = [nrmse] + list(nrmse_by_components)
        labels = ["Vector error"] + list(exp.variables_names)

        print(len(errors), len(labels))
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
        modals = ()
        if ytest is not None:
            modals, _, weights = gllim.modal_prediction(ytest, components=10, sort_by="weight")
            X = modals[0]
            modals = list(zip(X, weights[0]))
            ytest = ytest[0, 0]
        savepath = savepath or self.experience.archive.get_path("figures", filecategorie="schema")
        context = dict(**self.experience.get_infos(), max_Gamma=gllim.GammakList.max())
        self.G.schema_1D((x, y), gllim.ckList, gllim.ckListS, gllim.AkList, gllim.bkList, xlim, xtrue[0], ytest,
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

    def evolution_approx(self, thetas, savepath=None):
        """Show precision over iterations"""
        exp = self.experience
        savepath = savepath or exp.archive.get_path("figures", filecategorie="estimation-evo")
        l = []
        for theta in thetas:
            K = len(theta["pi"])
            gllim = exp.gllim_cls(K, Lw=exp.Lw, sigma_type=exp.sigma_type,
                                  gamma_type=exp.gamma_type, verbose=False)
            gllim.init_fit(exp.Xtrain, exp.Ytrain, theta)
            assert gllim.D == 1 and gllim.L == 1
            gllim.inversion()
            l.append(self.error_estimation(gllim, exp.Xtest))
        labels = self.LABELS_STUDY_ERROR
        self.G.simple_plot(list(zip(*l)), labels, "iterations", True, savepath=savepath,
                           title="Approximation evolution over iterations")


class VisualisationSecondLearning(MesuresSecondLearning, VisualisationMesures):

    def compare_density2D_parallel(self, Y, gllim_base, gllims, X=None, colorplot=True, savepath=None):
        exp = self.experience
        fsbefore, fsafter, modal_preds_before, modal_preds_after, trueXs = [], [], [], [], []

        nb_var = exp.partiel and len(exp.partiel) or len(exp.context.X_MAX)
        X = X if X is not None else [None] * len(Y)
        gllim_base.verbose = False
        for y, X0_obs, gllim in zip(Y, X, gllims):
            gllim.verbose = False
            Y0_obs = y[None, :]

            data = []

            Xweight, hs, ws = gllim_base.modal_prediction(Y0_obs, components=3)
            modal_pred_full_before = Xweight[0], hs[0], ws[0]

            Xweight, hs, ws = gllim.modal_prediction(Y0_obs, components=3)
            modal_pred_full_after = Xweight[0], hs[0], ws[0]

            def densitybefore_full(x_points, marginal, Y0_obs=Y0_obs):
                return gllim_base.forward_density(Y0_obs, x_points, marginals=marginal)

            def densityafter_full(x_points, marginal, gllim=gllim, Y0_obs=Y0_obs):
                return gllim.forward_density(Y0_obs, x_points, marginals=marginal)

            metadata = []
            for i in range(nb_var):
                for j in range(i + 1, nb_var):
                    densityb, xlim, ylim, modal_predb, trueX, varx, vary, titleb = \
                        self._collect_infos_density(densitybefore_full, "Density of {},{}", X0_obs, i,
                                                    j=j, modal_pred_full=modal_pred_full_before)

                    densitya, xlim, ylim, modal_preda, trueX, varx, vary, titlea = \
                        self._collect_infos_density(densityafter_full, "Snd learning - Density of {},{}", X0_obs, i,
                                                    j=j, modal_pred_full=modal_pred_full_after)

                    data.append((densityb, densitya, modal_predb, modal_preda, trueX))
                    metadata.append(((xlim, ylim), (varx, vary), titleb, titlea))

            fb, fa, mpb, mpa, tX = zip(*data)
            fsbefore.append(fb)
            fsafter.append(fa)
            modal_preds_before.append(mpb)
            modal_preds_after.append(mpa)
            trueXs.append(tX)
        varlims, varnames, titlesb, titlesa = zip(*metadata)
        savepath = savepath or exp.archive.get_path("figures", filecategorie="sequenceDensity2D")
        self.G.Density2DSequence(fsbefore, fsafter, varlims, varnames, titlesb, titlesa,
                                 modal_preds_before, modal_preds_after, trueXs, colorplot,
                                 savepath=savepath)
