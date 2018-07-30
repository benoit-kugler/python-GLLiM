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
from tools.experience import Experience

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

    graphiques.simple_plot(values, labels, None, True, title="Evolution de la log-vraisemblance",
                           savepath=PATHS[2])
    training.NB_MAX_ITER = old_MAXITER


def _train_K_N(exp, N_progression, K_progression):
    imax = len(N_progression)
    c = InjectiveFunction(1)(None)
    Xtest = c.get_X_sampling(10000)
    l = []
    X, Y = c.get_data_training(N_progression[-1])
    for i in range(imax):
        K = K_progression[i]
        N = N_progression[i]
        Xtrain = X[0:N, :]
        Ytrain = Y[0:N, :]
        # def ck_init_function():
        #     return c.get_X_uniform(K)
        logging.debug("\nFit {i}/{imax} for K={K}, N={N}".format(i=i + 1, imax=imax, K=K, N=Xtrain.shape[0]))
        gllim = training.multi_init(Xtrain, Ytrain, K, verbose=None)
        gllim.inversion()

        l.append(exp.mesures.error_estimation(gllim, Xtest))
    return np.array(l)


def plusieurs_K_N(imax):
    filename = "plusieursKN.mat"
    filename = os.path.join(Archive.BASE_PATH, filename)
    exp = Experience(InjectiveFunction(1))
    coeffNK = 10
    coeffmaxN1 = 10
    coeffmaxN2 = 2
    if RETRAIN:
        # k = 10 n
        K_progression = np.arange(imax) * 3 + 2
        N_progression = K_progression * coeffNK
        l1 = _train_K_N(exp, N_progression, K_progression)

        # N fixed
        K_progression = np.arange(imax) * 3 + 2
        N_progression1 = np.ones(imax, dtype=int) * (K_progression[-1] * coeffmaxN1)

        N_progression2 = np.ones(imax, dtype=int) * (K_progression[-1] * coeffmaxN2)

        l2 = _train_K_N(exp, N_progression1, K_progression)
        l3 = _train_K_N(exp, N_progression2, K_progression)

        scipy.io.savemat(filename + ".mat", {"l1": l1, "l2": l2, "l3": l3})
    else:
        m = scipy.io.loadmat(filename + ".mat")
        l1, l2, l3 = m["l1"], m["l2"], m["l3"]

    labels = exp.mesures.LABELS_STUDY_ERROR
    l1 = l1[:, 0]
    l2 = l2[:, 0]
    l3 = l3[:, 0]
    label1 = labels[0] + f" - $N = {coeffNK}K$"
    label2 = labels[0] + f" - $N = {coeffmaxN1} * Kmax$"
    label3 = labels[0] + f" - $N = {coeffmaxN2} * Kmax$"

    title = "Evolution de l'erreur en fonction de K et N"

    graphiques.plusieursKN([l1, l2, l3], [label1, label2, label3], None, True, savepath=PATHS[3],
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
    # Using comparaison trained model
    exp = Experience(context.LabContextOlivine, partiel=(0, 1, 2, 3), with_plot=True)
    exp.load_data(regenere_data=False, with_noise=50, N=10000, method="sobol")
    dGLLiM.dF_hook = exp.context.dF
    gllim = exp.load_model(200, mode="l", track_theta=False, init_local=200,
                           sigma_type="full", gamma_type="full", gllim_cls=jGLLiM)
    MCMC_X, Std = exp.context.get_result()
    exp.results.prediction_by_components(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                                         xtitle="wavelength (microns)", savepath=PATHS[9],
                                         Xref=MCMC_X, StdRef=Std, with_modal=2)

    exp.results.prediction_2D(gllim, exp.context.get_observations(), exp.context.wave_lengths,
                              Xref=MCMC_X, savepath=PATHS[10], xtitle="wavelength (microns)",
                              varlims=None, method="mean")


def main():
    plot_estimeF()
    plot_evo_LL()
    plusieurs_K_N(50)
    init_cos()
    regularization()
    comparaison_MCMC()


RETRAIN = False

if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
