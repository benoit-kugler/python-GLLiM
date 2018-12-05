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

import noise_GD
from noise_estimation import NoiseEstimation
from tools.context import LabContextOlivine, LabContextNontronite


def show_3D(*datas, labels=None):
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


def show_2D(*datas, labels=None, savepath=None, title=None):
    fig = pyplot.figure(figsize=(7,7))
    axe = fig.gca()
    D = len(datas)
    cm = matplotlib.cm.rainbow(np.linspace(0, 1, D))
    for i, data in enumerate(datas):
        color = cm[i]
        color[3] = 0.5
        label = labels[i] if labels is not None else f"Group {i}"
        axe.scatter(*data.T, c=color, label=label, s=2)
    axe.legend()
    if title:
        fig.suptitle(title)
    if savepath:
        pyplot.savefig(savepath)
    else:
        pyplot.show()


class Visualisation2D:

    BASE_SAVEPATH = "/scratch/WORK/NOISE_ESTIMATION/VISUALISATION"

    def __init__(self, observations, labels, fig_label = None):
        self.observations = observations
        """List of observations to compare."""

        self.labels = labels
        """List of corresponding labels"""

        self.fig_label = fig_label

    def _pca(self):
        ti = time.time()
        pca = PCA(n_components=2, copy=True)
        obs_ref = self.observations[0]
        pca.fit(obs_ref)
        res = [pca.transform(Y) for Y in self.observations]
        logging.debug(f"Principal Component Analysis performed in {time.time() - ti:.3f}")
        return res

    def _tsne(self):
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
        return res

    def reduction(self, methods=("pca", "tsne")):
        if type(methods) is str:
            methods = (methods, )

        if self.fig_label:
            path = os.path.join(self.BASE_SAVEPATH, self.fig_label)
        else:
            path = os.path.join(self.BASE_SAVEPATH, str(datetime.datetime.now()))

        if "pca" in methods:
            datas = self._pca()
            show_2D(*datas, labels = self.labels, savepath= path +  "-pca.png",
                    title=self.fig_label)

        if "tsne" in methods:
            datas = self._tsne()
            show_2D(*datas, labels = self.labels, savepath= path +  "-tsne.png",
                    title=self.fig_label)





def main():
    labels = ["Données d'entrainement","Données d'entrainement (bruit)", "Olivine", "Nontronite"]
    c = LabContextOlivine()
    Yobs1 = c.get_observations()

    c2 = LabContextNontronite()
    Yobs2 = c2.get_observations()

    _, Ytrain = c.get_data_training(50000)

    exp = NoiseEstimation(LabContextOlivine,"obs","diag","gd")
    noise_GD.maxIter = 150
    # exp.run_noise_estimator(save=True)
    mean, cov = exp.get_last_params(average_over=40)

    Ytrain_noise = c.add_noise_data(Ytrain,covariance=cov, mean=mean)

    visu = Visualisation2D((Ytrain, Ytrain_noise, Yobs1, Yobs2), labels, fig_label="Bruit GD")
    visu.reduction(methods=("pca","tsne"))


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # show_diff_learning()
    # tsne()
    main()