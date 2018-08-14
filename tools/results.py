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
    Xweight = np.array([xs[ps] for ps, xs in zip(perms, Xweight)])
    logging.debug(f"Done in  {time.time() - t} s")
    return Xweight


class Results():

    NB_MODAL_PREDS = 2
    MODAL_PREDS_ORDER = "weight"

    def __init__(self, experience):
        self.experience = experience

    def full_prediction(self, gllim, Y, with_regu=True, with_modal=3):
        exp = self.experience
        if with_modal:
            Xweight, heights, weights = gllim.modal_prediction(Y, components=with_modal, sort_by="weight")
            Xweight = np.array(Xweight)
            if with_regu:
                Xweight = _modal_regularization(exp.context.normalize_X, "exclu", Xweight)
        else:
            Xweight, heights, weights = None, None, None
        Xmean, Covs = gllim.predict_high_low(Y, with_covariance=True)
        return Xmean, Covs, Xweight, heights, weights


class VisualisationResults(Results):

    def __init__(self, experience):
        super(VisualisationResults, self).__init__(experience)
        from plotting import graphiques
        self.G = graphiques

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

            self.G.ModalPred1D(Xs, Xm, xlabels, varlim, varnames, "Poids de rang {}", context=exp.get_infos(),
                               write_context=True,
                               title="Prédiction modale - Vue par composants", savepath=savepath)

            self.G.ModalPred1D(Xs1, Xm, xlabels, varlim, varnames, "Initialisation avec poids de rang {}",
                               context=exp.get_infos(),
                               write_context=False, savepath=savepath1,
                               title="Prédiction modale - Vue par composants - Avec régularisation par permutation")

            self.G.ModalPred1D(Xs2, Xm, xlabels, varlim, varnames, "Initialisation avec poids de rang {}",
                               context=exp.get_infos(), write_context=False,
                               title="Prédiction modale - Vue par composants - Avec régularisation par fusion",
                               savepath=savepath2)

    def prediction_by_components(self, gllim: GLLiM, Y, labels, xtitle="observations", varlims=None, with_modal=False,
                                 savepath=None, Xref=None, StdRef=None, with_regu=True, indexes=None):
        """Draw one axe by variable, with optionnal reference, standard deviation,
        and modal predictions (with exlu regularization). Return mean predictions with covariances"""
        exp = self.experience
        savepath = savepath or exp.archive.get_path("figures", filecategorie="synthese1D")
        Xmean, Covs, Xweight, _, _ = self.full_prediction(gllim, Y, with_modal=with_modal, with_regu=with_regu)
        varlims = exp.variables_lims if (varlims == "context") else varlims
        varnames = exp.variables_names
        if indexes is not None:
            Xmean = Xmean[:, indexes]
            Covs = Covs[:, indexes][:, :, indexes]
            Xweight = Xweight[:, :, indexes]
            Xref = Xref[:, indexes] if Xref is not None else None
            StdRef = StdRef[:, indexes] if StdRef is not None else None
            varlims = varlims[list(indexes)] if varlims is not None else None
            varnames = varnames[list(indexes)]
        self.G.Results_1D(Xmean, Covs, Xweight, labels, xtitle, varnames,
                          varlims, Xref, StdRef, context=exp.get_infos(Ntest="-"),
                          title="Prédiction - Vue par composants",
                          savepath=savepath, write_context=True)
        return Xmean, Covs

    def prediction_2D(self, gllim: GLLiM, Y, labels_value, xtitle="observations", method="mean",
                      varlims=None, Xref=None, savepath=None):
        """Prediction for each Y and plot 2D with labels as color.
        If method is weight or height or bestY, use best modal prediction"""
        exp = self.experience
        X = exp._one_X_prediction(gllim, Y, method)
        varlims = exp.variables_lims if (varlims == "context") else varlims
        varnames = exp.variables_names
        savepath = savepath or exp.archive.get_path("figures", filecategorie="correlations-{}".format(method))
        self.G.Results_2D(X, labels_value, xtitle, varnames, varlims, Xref, context=exp.get_infos(Ntest="-"),
                          title="Corrélations - Mode de prédiction :  {}".format(method), write_context=True,
                          savepath=savepath)

    def plot_density_sequence(self, gllim: GLLiM, Y, xlabels, index=0, varlims=None, xtitle="observations",
                              Xref=None, StdRef=None, with_pdf_images=False, regul=None, post_processing=None):
        Xmean, Covs, Xweight, heights, weights = self.full_prediction(gllim, Y, with_regu=regul)
        StdMean = np.sqrt(Covs[:, index, index])
        exp = self.experience

        def densitys(x_points):
            return gllim.forward_density(Y, x_points, marginals=(index,))

        modal_preds = []
        for y, xs, hs, ws in zip(Y, Xweight, heights, weights):
            if post_processing:
                xs = post_processing(xs)
            mpred = list(zip(xs[:, index], hs, ws))
            modal_preds.append(mpred)
        xlim = varlims or exp.variables_lims[index]
        varname = exp.variables_names[index]

        if with_pdf_images:
            pdf_paths = exp.context.get_images_path_densities(index)
        else:
            pdf_paths = None

        if Xref is not None:
            Xref = Xref[:, index]

        if StdRef is not None:
            StdRef = StdRef[:, index]

        if post_processing:
            Xweight = np.array([post_processing(X) for X in Xweight])
            Xmean = np.array([post_processing(X) for X in Xmean])

        self.G.density_sequences1D(densitys, modal_preds, xlabels, xtitle, Xmean[:, index], Xweight[:, :, index],
                                   xlim, varname, Xref, StdRef, StdMean, pdf_paths,
                                   title="Densities - {}".format(varname),
                                   savepath=exp.archive.get_path("figures", filecategorie="sequence"))

    def map(self, gllim: GLLiM, Y, latlong, index, Xref=None, savepath=None):
        X = gllim.predict_high_low(Y)
        x = X[:, index]
        varname = self.experience.variables_names[index]
        if Xref is not None:
            Xref = Xref[:, index]
        savepath = savepath or self.experience.archive.get_path("figures",
                                                                filecategorie="map-{}".format(varname))
        self.G.MapValues(latlong, x, Xref, ("GLLiM", "MCMC"),
                         title="Parameter {}".format(varname), savepath=savepath)
