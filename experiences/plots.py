"""Trains gllim  and plot severals graphs"""
import logging
import os

import coloredlogs
import numpy as np
import scipy.io
from PIL import Image

from Core import training
from Core.dgllim import dGLLiM
from Core.gllim import GLLiM, jGLLiM
from plotting import graphiques
from tools import context
from tools.archive import Archive
from tools.context import WaveFunction, InjectiveFunction, HapkeContext
from tools.experience import Experience, mesure_convergence, Ntest_PLUSIEURS_KN
from tools.measures import Mesures

LATEX_IMAGES_PATH = "../latex/images/plots"


def PATHS(s):
    return os.path.join(LATEX_IMAGES_PATH, s)


def _merge_image_byside(paths, savepath, remove=False):
    """If remove is given, remove images in paths after merging"""
    widths, heights = zip(*(i.size for i in map(Image.open, paths)))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), color="white")

    x_offset = 0
    for im in map(Image.open, paths):
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(savepath)

    if remove:
        for path in paths: os.remove(path)


def exemple_pre_lissage():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)

    exp.load_data(regenere_data=False, with_noise=50, N=100000, method="sobol")
    gllim = exp.load_model(100, mode="l", track_theta=False, init_local=100,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    exp.results.prediction_by_components(gllim, exp.context.get_observations(),
                                         exp.context.wavelengths, with_modal=3, indexes=(0,),
                                         with_regu=False, xtitle="longeur d'onde ($\mu$m)",
                                         savepath=PATHS("pre-lissage.png"))


def plot_estimeF_simple():
    exp = Experience(context.ExampleFunction, partiel=None, with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000, method="sobol")

    gllim = exp.load_model(30, mode=RETRAIN and "r" or "l", track_theta=False, init_local=100,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    p1 = PATHS("estimFsimple_tmp1.png")
    p2 = PATHS("estimFsimple_tmp2.png")
    exp.mesures.plot_estimatedF(gllim, [0], savepath=p1,
                                title=f"Estimation de F : {exp.context.LABEL}",
                                write_context=False)
    exp.mesures.plot_clusters(gllim, savepath=p2, draw_context=False, write_context=False,
                              title="Découpage de l'espace de départ")
    p = PATHS("estimFsimple.png")
    _merge_image_byside((p2, p1), p, remove=True)


def plot_estimeF():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1), with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000, method="sobol")

    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    p1 = PATHS("estimF1.png")
    var = f"({exp.variables_names[0]} , {exp.variables_names[1]})"
    exp.mesures.plot_estimatedF(gllim, [0, 2, 4, 8], savepath=p1,
                                title=f"Estimation de $F_{{hapke}}$ - variables {var}",
                                write_context=True)
    #
    #
    exp = Experience(context.LabContextOlivine, partiel=(2, 3), with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000, method="sobol")

    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    p2 = PATHS("estimF2.png")
    var = f"({exp.variables_names[0]} , {exp.variables_names[1]})"
    exp.mesures.plot_estimatedF(gllim, [0, 2, 4, 8], savepath=p2,
                                title=f"Estimation de $F_{{hapke}}$ - variables {var}",
                                write_context=True)

    # merging both
    # _merge_image_byside([p1,p2],PATHS[0])
    # graphiques.estimated_F.write_context(exp.get_infos(),PATHS[0])


def plot_evo_LL():
    training.NB_MAX_ITER, old_MAXITER = 200, training.NB_MAX_ITER
    values, labels = [], []
    exp = Experience(WaveFunction, partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=1000)
    gllim = exp.load_model(20, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=GLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    exp = Experience(InjectiveFunction(4), partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=2000)
    gllim = exp.load_model(50, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=GLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    exp = Experience(HapkeContext, partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000)
    gllim = exp.load_model(50, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=jGLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    graphiques.simple_plot(values, labels, None, True, "Itérations", "Log-vraisemblance",
                           title="Evolution de la log-vraisemblance",
                           savepath=PATHS("evoLL1.png"))
    training.NB_MAX_ITER = old_MAXITER





def plusieurs_K_N(imax):
    l1, l2, l3, K_progression, coeffNK, coeffmaxN1, coeffmaxN2 = mesure_convergence(imax, RETRAIN)
    labels = Mesures.LABELS_STUDY_ERROR
    l1 = l1[:, 0]
    l2 = l2[:, 0]
    l3 = l3[:, 0]
    label1 = "1 - " + labels[0] + f" - $N = {coeffNK}K$"
    label2 = "2 - " + labels[0] + f" - $N = {coeffmaxN1} * Kmax$"
    label3 = "3 - " + labels[0] + f" - $N = {coeffmaxN2} * Kmax$"

    title = "Evolution de l'erreur en fonction de K et N"
    xlabels = K_progression
    graphiques.plusieursKN([l1, l2, l3], [label1, label2, label3], xlabels, True, "K", "Erreur moyenne",
                           savepath=PATHS("evoKN.png"),
                           title=title, write_context=True,
                           context={"coeffNK": coeffNK, "coeffmaxN1": coeffmaxN1, "coeffmaxN2": coeffmaxN2,
                                    "Ntest": Ntest_PLUSIEURS_KN})


def init_cos():
    exp = Experience(WaveFunction, partiel=None, verbose=None, with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000)

    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=None,
                           gamma_type="full", gllim_cls=GLLiM)

    # assert np.allclose(gllim.AkList, np.array([exp.context.dF(c) for c in gllim.ckList]))

    x = np.array([0.55])
    y = exp.context.F(x[None, :])
    exp.mesures.illustration(gllim, x, None, savepath=PATHS("init_cos1.png"))

    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=100,
                           gamma_type="full", gllim_cls=GLLiM)

    exp.mesures.illustration(gllim, x, y, savepath=PATHS("init_cos2.png"))


def regularization():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=50, N=100000, method="sobol")
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    filenames = [PATHS("modalPred1.png"), None, None, None]
    filenames_regu = [PATHS("modalPred2.png"), None, None, None]
    filenames_regu2 = [PATHS("modalPred3.png"), None, None, None]
    exp.results.plot_modal_preds_1D(gllim, exp.context.get_observations(), exp.context.wavelengths,
                                    varlims=[(0, 1), (-0.5, 0.5), (0, 30), (0.5, 1.2)],
                                    filenames=filenames, filenames_regu=filenames_regu,
                                    filenames_regu2=filenames_regu2)


# X = exp.best_Y_prediction(gllim,exp.context.get_observations())


def comparaison_MCMC():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=50, N=100000, method="sobol")
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)
    MCMC_X, Std = exp.context.get_result()
    # exp.results.prediction_by_components(gllim, exp.context.get_observations(), exp.context.wavelengths,
    #                                      xtitle="wavelength (microns)", savepath=PATHS("results1.png"),
    #                                      Xref=MCMC_X, StdRef=Std, with_modal=2)

    exp.results.prediction_2D(gllim, exp.context.get_observations(), exp.context.wavelengths,
                              Xref=MCMC_X, savepath=PATHS("results2.png"), xtitle="wavelength (microns)",
                              varlims=None, method="mean", indexes=[0, 1, 3])


def plot_sol_multiples():
    """Learning has been made by comparaison_MCMC"""
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)

    exp.load_data(regenere_data=False, with_noise=50, N=100000, method="sobol")
    gllim = exp.load_model(100, mode="l", track_theta=False, init_local=100,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    n = 16
    Y0_obs, X0_obs = exp.Ytest[n:n + 1], exp.Xtest[n]
    exp.mesures.plot_conditionnal_density(gllim, Y0_obs, X0_obs, with_modal=2, colorplot=False, write_context=True,
                                          draw_context=False, savepath=PATHS("solmult2D.png"))
    exp.mesures.plot_conditionnal_density(gllim, Y0_obs, X0_obs, with_modal=2, dim=1, write_context=True,
                                          savepath=PATHS("solmult1D.png"))


def plot_map():
    exp = Experience(context.HapkeContext, partiel=None, with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=50, N=50000, method="sobol")
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           gllim_cls=jGLLiM)

    Y = exp.context.get_observations()
    latlong, mask = exp.context.get_spatial_coord()
    Y = Y[mask]  # cleaning
    MCMC_X, Std = exp.context.get_result(with_std=True)
    MCMC_X = MCMC_X[mask]

    exp.results.map(gllim, Y, latlong, 0, Xref=MCMC_X, savepath=PATHS("map.png"))
    diff = gllim.predict_high_low(Y)[:, 0] - MCMC_X[:, 0]
    exp.results.G.MapValues(latlong, diff, None, ("Différence GLLiM - MCMC",), savepath=PATHS("map-diff.png"),
                            custom_context="Moyenne {0:.3f}, écart-type {1:.3f}".format(diff.mean(), diff.std()),
                            write_context=True)

def main():
    # exemple_pre_lissage()
    # plot_estimeF_simple()
    # plot_estimeF()
    # plot_evo_LL()
    # plusieurs_K_N(20)
    # init_cos()

    # regularization()
    # comparaison_MCMC()
    # plot_sol_multiples()
    plot_map()


RETRAIN = False

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
