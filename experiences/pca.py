import datetime
import logging
import os
import time

import coloredlogs
import matplotlib
import matplotlib.cm
import numpy as np
import vispy.scene
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from vispy.scene import visuals

from Core import noise_GD
from experiences.noise_estimation import NoiseEstimation
from tools.context import LabContextOlivine, LabContextNontronite, MergedLabObservations


def draw_3D(*datas, labels=None):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor="white")
    view = canvas.central_widget.add_view()

    pos, colors = [], []
    D = len(datas)
    cm = matplotlib.cm.rainbow(np.linspace(0, 1, D))
    for i, data in enumerate(datas):
        pos.extend(data.T)
        color = cm[i]
        colors.extend([color] * len(data.T))

    colors = np.array(colors)
    pos = np.array(pos)

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color=None, face_color=colors, size=5)

    view.add(scatter)

    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    view.add(axis)

    if labels is not None:  # legend
        for i, label in enumerate(labels):
            leg = visuals.Text(label, color=cm[i], pos=(0, 10 * i))
            canvas.scene.add(leg)


    return canvas


def draw_2D(*datas, labels=None, savepath=None, title=None):
    fig = pyplot.figure(figsize=(7,7))
    axe = fig.gca()
    D = len(datas)
    cm = matplotlib.cm.rainbow(np.linspace(0, 1, D))
    for i, data in enumerate(datas):
        color = cm[i]
        color[3] = 0.5
        label = labels[i] if labels is not None else f"Group {i}"
        axe.scatter(*data.T, c=[color], label=label, s=2)
    axe.legend()
    if title:
        fig.suptitle(title)
    if savepath:
        pyplot.savefig(savepath)
    else:
        pyplot.show()


class Visualisation2D:

    BASE_SAVEPATH = "/Users/kuglerb/Documents/WORK/NOISE_ESTIMATION/VISUALISATION"

    def __init__(self, observations, labels, fig_label = None):
        self.observations = observations
        """List of observations to compare."""

        self.labels = labels
        """List of corresponding labels"""

        self.fig_label = fig_label

    def pca(self, index_ref=0):
        ti = time.time()
        pca = PCA(n_components=2, copy=True)
        obs_ref = self.observations[index_ref]
        pca.fit(obs_ref)
        res = [pca.transform(Y) for Y in self.observations]
        logging.debug(f"Principal Component Analysis performed in {time.time() - ti:.3f}")
        self.reductions = res
        self.method = "pca"

    def tsne(self):
        ti = time.time()

        group = np.concatenate(self.observations, axis=0)
        tsne = TSNE(n_components=2)
        visu = tsne.fit_transform(group)

        lengths = [len(Y) for Y in self.observations]
        i = 0
        res = []
        for l in lengths:
            Z = visu[i:i+l]
            i = i+l+1
            res.append(Z)

        logging.info(f"Stochastic Neighbor Embedding performed in {time.time() - ti:.3f} s")
        self.reductions = res
        self.method = "tsne"


    def draw(self, partiel = None):
        if self.fig_label:
            path = os.path.join(self.BASE_SAVEPATH, self.fig_label)
        else:
            path = os.path.join(self.BASE_SAVEPATH, str(datetime.datetime.now()))

        if partiel is not None:
            datas = [self.reductions[i] for i in partiel]
            labels = [self.labels[i] for i in partiel]
            path += str(partiel)
        else:
            datas, labels = self.reductions, self.labels

        savepath = f"{path}-{self.method}.png"
        draw_2D(*datas, labels = labels, savepath= savepath,
                title=self.fig_label)
        logging.debug(f"Figure saved in {savepath}.")



def main(retrain=True):
    labels = ["Yobs1", "Yobs2", "Entrainement","Entrainement (bruit Olivine)",
              "Entrainement (bruit Nontronite)", "Entrainement (bruit mélangé)"]
    c = LabContextOlivine()
    Yobs1 = c.get_observations()

    c2 = LabContextNontronite()
    Yobs2 = c2.get_observations()

    _, Ytrain = c.get_data_training(5000)


    noise_GD.Ntrain = 100000
    noise_GD.maxIter = 100

    exp1 = NoiseEstimation(LabContextOlivine,"obs","diag","gd")
    exp2 = NoiseEstimation(LabContextNontronite,"obs","diag","gd")
    exp3 = NoiseEstimation(MergedLabObservations,"obs","diag","gd")

    if retrain:
        exp1.run_noise_estimator(save=True)
        exp2.run_noise_estimator(save=True)
        exp3.run_noise_estimator(save=True)

    mean1, cov1 = exp1.get_last_params(average_over=40)
    mean2, cov2 = exp2.get_last_params(average_over=40)
    mean3, cov3 = exp3.get_last_params(average_over=40)


    Ytrain_noise1 = c.add_noise_data(Ytrain,covariance=cov1, mean=mean1)
    Ytrain_noise2 = c.add_noise_data(Ytrain,covariance=cov2, mean=mean2)
    Ytrain_noise3 = c.add_noise_data(Ytrain,covariance=cov3, mean=mean3)

    visu = Visualisation2D((Yobs1, Yobs2, Ytrain, Ytrain_noise1, Ytrain_noise2, Ytrain_noise3), labels, fig_label="Bruit GD")
    visu.pca(index_ref=2)
    visu.draw()
    for i in range(4):
        visu.draw(partiel=(0,1, i + 2))


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # show_diff_learning()
    # tsne()
    main(retrain=False)