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
from tools.experience import Experience, mesure_convergence
from tools.measures import Mesures

LATEX_IMAGES_PATH = "../latex/images/plots"

names = ["estimF1.png", "estimF2.png", "evoLL1.png", "evoKN.png",
         "init_cos1.png", "init_cos2.png",
         "modalPred1.png", "modalPred2.png", "modalPred3.png",
         "results1.png", "results2.png"]
PATHS = [os.path.join(LATEX_IMAGES_PATH, i) for i in names]



def _merge_image_byside(paths, savepath):
    widths, heights = zip(*(i.size for i in map(Image.open, paths)))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in map(Image.open, paths):
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(savepath)


def plot_estimeF():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1), with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000, method="sobol")

    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    p1 = PATHS[0]
    var = f"$({exp.variables_names[0]} , {exp.variables_names[1]})$"
    exp.mesures.plot_estimatedF(gllim, [0, 2, 4, 8], savepath=p1, title=f"Estimation de F - variables {var}",
                                write_context=True)
    #
    #
    exp = Experience(context.LabContextOlivine, partiel=(2, 3), with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000, method="sobol")

    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)

    p2 = PATHS[1]
    var = f"$({exp.variables_names[0]} , {exp.variables_names[1]})$"
    exp.mesures.plot_estimatedF(gllim, [0, 2, 4, 8], savepath=p2, title=f"Estimation de F - variables {var}",
                                write_context=True)

    # merging both
    # _merge_image_byside([p1,p2],PATHS[0])
    # graphiques.estimated_F.write_context(exp.get_infos(),PATHS[0])


def plot_evo_LL():
    training.NB_MAX_ITER, old_MAXITER = 200, training.NB_MAX_ITER
    values, labels = [], []
    exp = Experience(WaveFunction, partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000)
    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=GLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    exp = Experience(InjectiveFunction(4), partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=800)
    gllim = exp.load_model(10, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=GLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    exp = Experience(HapkeContext, partiel=None, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=1000)
    gllim = exp.load_model(10, mode=RETRAIN and "r" or "l", track_theta=True, init_local=200,
                           gamma_type="full", gllim_cls=GLLiM)

    _, LLs = exp.archive.load_tracked_thetas()
    LLs = (np.array(LLs[1:]) - LLs[0]) / (exp.context.D + exp.context.L)
    values.append(LLs)
    labels.append(exp.context.LABEL)

    graphiques.simple_plot(values, labels, None, True, "Itérations", "Log-vraisemblance",
                           title="Evolution de la log-vraisemblance",
                           savepath=PATHS[2])
    training.NB_MAX_ITER = old_MAXITER





def plusieurs_K_N(imax):
    l1, l2, l3, K_progression, coeffNK, coeffmaxN1, coeffmaxN2 = mesure_convergence(imax, RETRAIN)

    labels = Mesures.LABELS_STUDY_ERROR
    l1 = l1[:, 0]
    l2 = l2[:, 0]
    l3 = l3[:, 0]
    label1 = labels[0] + f" - $N = {coeffNK}K$"
    label2 = labels[0] + f" - $N = {coeffmaxN1} * Kmax$"
    label3 = labels[0] + f" - $N = {coeffmaxN2} * Kmax$"

    title = "Evolution de l'erreur en fonction de K et N"
    xlabels = K_progression
    graphiques.plusieursKN([l1, l2, l3], [label1, label2, label3], xlabels, True, "K", "Erreur", savepath=PATHS[3],
                           title=title, write_context=True,
                           context={"coeffNK": coeffNK, "coeffmaxN1": coeffmaxN1, "coeffmaxN2": coeffmaxN2})


def init_cos():
    exp = Experience(WaveFunction, partiel=None, verbose=None, with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=None, N=10000)

    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=None,
                           gamma_type="full", gllim_cls=GLLiM)

    # assert np.allclose(gllim.AkList, np.array([exp.context.dF(c) for c in gllim.ckList]))

    x = np.array([0.55])
    y = exp.context.F(x[None, :])
    exp.mesures.illustration(gllim, x, y, savepath=PATHS[4])

    gllim = exp.load_model(100, mode=RETRAIN and "r" or "l", track_theta=False, init_local=100,
                           gamma_type="full", gllim_cls=GLLiM)

    exp.mesures.illustration(gllim, x, y, savepath=PATHS[5])


def regularization():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True, verbose=None)
    exp.load_data(regenere_data=RETRAIN, with_noise=50, N=10000, method="latin")
    dGLLiM.dF_hook = exp.context.dF
    # X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
    gllim = exp.load_model(1000, mode=RETRAIN and "r" or "l", track_theta=False, init_local=500,
                           sigma_type="iso", gamma_type="full", gllim_cls=dGLLiM)

    filenames = [PATHS[6], None, None, None]
    filenames_regu = [PATHS[7], None, None, None]
    filenames_regu2 = [PATHS[8], None, None, None]
    exp.results.plot_modal_preds_1D(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                                    varlims=[(0, 1), (-0.5, 0.5), (0, 30), (0.5, 1.2)],
                                    filenames=filenames, filenames_regu=filenames_regu,
                                    filenames_regu2=filenames_regu2)


# X = exp.best_Y_prediction(gllim,exp.context.get_observations())


def comparaison_MCMC():
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)
    exp.load_data(regenere_data=RETRAIN, with_noise=50, N=100000, method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    gllim = exp.load_model(150, mode=RETRAIN and "r" or "l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)
    MCMC_X, Std = exp.context.get_result()
    exp.results.prediction_by_components(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                                         xtitle="wavelength (microns)", savepath=PATHS[9],
                                         Xref=MCMC_X, StdRef=Std, with_modal=2)

    exp.results.prediction_2D(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                              Xref=MCMC_X, savepath=PATHS[10], xtitle="wavelength (microns)",
                              varlims=None, method="mean")


def main():
    # plot_estimeF()
    # plot_evo_LL()
    plusieurs_K_N(20)
    # init_cos()
    # regularization()
    # comparaison_MCMC()


RETRAIN = False

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
