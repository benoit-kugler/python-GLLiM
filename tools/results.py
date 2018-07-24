"""Implements tools to visualize results from GLLiM (mean and modal). Includes use of regularization."""
import logging
import time

import numpy as np

from Core.gllim import GLLiM
from tools import regularization


def _modal_regularization(renormalization, mode, Xweight):
    reg_f = {"step": regularization.step_by_step,
             "exclu": regularization.global_regularization_exclusion}[mode]
    logging.debug("Regularization {} ... ".format(mode))
    t = time.time()
    Xnorm = renormalization(Xweight)
    perms = reg_f(Xnorm)
    Xweight2 = np.array([xs[ps] for ps, xs in zip(perms, Xweight)])
    assert Xweight2.shape == Xweight.shape
    logging.debug("Done in ", time.time() - t, "s")
    return Xweight2


class Results():
    NB_MODAL_PREDS = 2
    MODAL_PREDS_ORDER = "weight"

    def __init__(self, experience):
        self.experience = experience
        from plotting import graphiques
        self.G = graphiques


class VisualisationResults(Results):

    def plot_modal_preds_1D(self, gllim: GLLiM, Y, xlabels, varlims=None,
                            filenames=None, filenames_regu=None, filenames_regu2=None):
        """"Plot modal preds without regularization then with regularization, one parameter by figure.
        For academic purpose."""
        exp = self.experience
        varlims = varlims or exp.variables_lims
        basepath = exp.archive.get_path("figures", filecategorie="synthese1D")
        savepaths = filenames or [basepath[:-3] + v + ".png" for v in exp.variables_names]

        Xweight, _, _ = gllim.modal_prediction(Y, components=self.NB_MODAL_PREDS,
                                               sort_by=self.MODAL_PREDS_ORDER)
        Xweight = np.array(Xweight)
        Xmean = gllim.predict_high_low(Y)
        renormalization = exp.context.normalize_X
        XweightR1 = _modal_regularization(renormalization, "exclu", Xweight)
        basepath = exp.archive.get_path("figures", filecategorie="synthese1D-regu")
        savepathsR1 = filenames_regu or [basepath[:-4] + v + ".png" for v in exp.variables_names]

        XweightR2 = _modal_regularization(renormalization, "step", Xweight)
        basepath = exp.archive.get_path("figures", filecategorie="synthese1D-reguglobal")
        savepathsR2 = filenames_regu2 or [basepath[:-4] + v + ".png" for v in exp.variables_names]
        for i in range(len(exp.variables_lims)):
            Xs, Xs1, Xs2, = Xweight[:, :, i], XweightR1[:, :, i], XweightR2[:, :, i]
            Xm = Xmean[:, i]
            varlim, varnames = varlims[i], exp.variables_names[i]
            savepath, savepath1, savepath2 = savepaths[i], savepathsR1[i], savepathsR2[i]

            self.G.ModalPred1D(Xs, Xm, xlabels, varlim, varnames, context=exp.get_infos(), write_context=True,
                               title="Prédiction modale - Vue par composants", savepath=savepath)

            self.G.ModalPred1D(Xs1, Xm, xlabels, varlim, varnames, context=exp.get_infos(), write_context=True,
                               title="Prédiction modale - Vue par composants - Avec régularisation par permutation",
                               savepath=savepath1)

            self.G.ModalPred1D(Xs2, Xm, xlabels, varlim, varnames, context=exp.get_infos(), write_context=True,
                               title="Prédiction modale - Vue par composants - Avec régularisation par fusion",
                               savepath=savepath2)

    def plot_correlations2D(self, gllim: GLLiM, Y, labels_value, method="mean",
                            varlims=None, add_points=None):
        """Prediction for each Y and plot 2D with labels as color"""
        X = self.experience._one_X_prediction(gllim, Y, method)
        varlims = varlims or self.experience.variables_lims
        varnames = self.experience.variables_names
        correlations2D(X, labels_value, self.experience.get_infos(), varnames, varlims,
                       main_title="Corrélations - Prediction mode :  {}".format(method), add_points=add_points,
                       savepath=self.experience.archive.get_path("figures",
                                                                 filecategorie="correlations-{}".format(method)))

    def prediction_by_components(self, gllim: GLLiM, Y, labels, varlims=None, regul=None, filename=None):
        exp = self.experience
        varlims = varlims or exp.variables_lims
        savepath = exp.archive.get_path("figures", filecategorie="synthese1D", filename=filename)
        Xweight, heights, weights = gllim.modal_prediction(Y, components=3, sort_by="weight")
        Xweight = np.array(Xweight)
        if regul:
            Xweight, _ = _modal_regularization(regul, Xweight)
        Xmean, Covs = gllim.predict_high_low(Y, with_covariance=True)
        correlations1D(Xmean, Xweight, Covs, labels, exp.get_infos(), exp.variables_names,
                       varlims, main_title="Prédiction - Vue par composants",
                       savepath=savepath)
        return Xmean, Covs

    def plot_density_sequence(self, gllim: GLLiM, Y, labels_value, index=0, varlims=None,
                              Xref=None, StdRef=None, with_pdf_images=False, regul=None, post_processing=None):
        Xs, heights, weights = gllim.modal_prediction(Y, components=None)
        Xweight, Xheight = [], []
        for xs, ws in zip(Xs, weights):
            Xheight.append(xs[0:3])
            l = zip(xs, ws)
            l = sorted(l, key=lambda d: d[1], reverse=True)[0:3]
            Xweight.append([x[0] for x in l])
        Xweight = np.array(Xweight)
        Xheight = np.array(Xheight)

        if regul:
            Xweight, Xheight = _modal_regularization(regul, Xweight, Xheight)

        self._plot_density_sequence(gllim, Y, labels_value, Xweight, Xheight, Xref, StdRef, with_pdf_images,
                                    index, threshold=0.001, varlims=varlims, post_processing=post_processing)

    def plot_density_sequence_clustered(self, gllim, Y, Xw_clus, Xh_clus, labels_value, index=0, varlims=None,
                                        Xref=None, StdRef=None, with_pdf_images=False):
        Xweight = np.array([xs[0:2, :] for xs in Xw_clus])
        Xheight = np.array([xs[0:2, :] for xs in Xh_clus])

        self._plot_density_sequence(gllim, Y, labels_value, Xweight, Xheight, Xref, StdRef, with_pdf_images,
                                    index, threshold=0.001, varlims=varlims)

    def _plot_density_sequence(self, gllim, Y, labels_x, Xweight, Xheight, Xref, StdRef,
                               with_pdf_images, index, threshold=0.01, varlims=None, post_processing=None):
        exp = self.experience
        fs, xlims, ylims, modal_preds, trueXs, varnames, titles = [], [], [], [], [], [], []

        Xmean, Covs = gllim.predict_high_low(Y, with_covariance=True)
        StdMean = np.sqrt(Covs[:, index, index])

        Xs, heights, weights = gllim.modal_prediction(Y, components=None)
        for y, xs, hs, ws in zip(Y, Xs, heights, weights):
            Y0_obs = y[None, :]

            def density(x_points, Y0_obs=Y0_obs):
                return gllim.forward_density(Y0_obs, x_points, marginals=(index,))

            xs = np.array([x for x, w in zip(xs, ws) if w >= threshold])
            if post_processing:
                xs = post_processing(xs)
            mpred = list(zip(xs[:, index], hs, ws))
            fs.append(density)
            modal_preds.append(mpred)
        xlim = varlims or exp.variables_lims[index]
        xvar = exp.variables_names[index]

        if with_pdf_images:
            pdf_paths = exp.context.get_images_path_densities(index)
        else:
            pdf_paths = None

        if Xref is not None:
            Xref = Xref[:, index]

        if StdRef is not None:
            StdRef = StdRef[:, index]

        if post_processing:
            Xheight = np.array([post_processing(X) for X in Xheight])
            Xweight = np.array([post_processing(X) for X in Xweight])
            Xmean = np.array([post_processing(X) for X in Xmean])

        density_sequences1D(fs, modal_preds, labels_x, Xmean[:, index], Xweight[:, :, index], Xheight[:, :, index],
                            StdMean=StdMean,
                            Yref=Xref, StdRef=StdRef,
                            title="Densities - ${}$".format(xvar), xlim=xlim, images_paths=pdf_paths,
                            savepath=exp.archive.get_path("figures", filecategorie="sequence"))

    def map(self, gllim: GLLiM, Y, latlong, index, Xref=None):
        X = gllim.predict_high_low(Y)
        x = X[:, index]
        varname = self.experience.variables_names[index]
        if Xref is not None:
            Xref = Xref[:, index]

        map_values(latlong, x, addvalues=Xref, main_title="Parameter ${}$".format(varname),
                   titles=("GLLiM", "MCMC"),
                   savepath=self.experience.archive.get_path("figures", filecategorie="map-{}".format(varname)))
