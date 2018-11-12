import logging
import time

import coloredlogs
import matplotlib.cm
import matplotlib
import numpy as np
import vispy.scene

from tools.context import LabContextOlivine, LabContextNontronite

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot
from vispy import scene, app, color
from vispy.scene import visuals

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiences import rtls


def show_diff_learning():
    h = LabContextOlivine()
    Yobs = h.get_observations()

    Yobs2 = LabContextNontronite().get_observations()

    _, Y = h.get_data_training(100000)

    # Ynoise20 = h.add_noise_data(Y, precision=20)
    # Ynoise50 = h.add_noise_data(Y, precision=50)
    #
    # d1 = np.mean([np.linalg.norm(Y - yobs, axis=1).min() for yobs in Yobs])
    # d2 = np.mean([np.linalg.norm(Ynoise20 - yobs, axis=1).min() for yobs in Yobs])
    # d3 = np.mean([np.linalg.norm(Ynoise50 - yobs, axis=1).min() for yobs in Yobs])
    #
    # print(d1, d2, d3)
    #
    # print(np.abs(Y - Ynoise20).mean())
    # print(np.abs(Y - Ynoise50).mean())
    # return

    pca = PCA(n_components=2, copy=True)

    pca.fit(Y)

    Z = pca.transform(Y)
    Zobs = pca.transform(Yobs)
    Zobs2 = pca.transform(Yobs2)
    # Z20 = pca.transform(Ynoise20)
    # Z50 = pca.transform(Ynoise50)

    show_2D(Z, Zobs, Zobs2, labels=["Entrainement", "Olivine", "Nontronite"],
            savepath="/home/bkugler/Documents/reunion8_11/pca.png")

    # fig = pyplot.figure()
    # fig.add_subplot(111, projection="3d")
    # axe = fig.gca()
    # axe.scatter(*Z.T, label="train 0")
    # axe.scatter(*Zobs.T, label="obs")
    # axe.scatter(*Z20.T, label="train 1/20")
    # axe.scatter(*Z50.T, label="train 1/50")
    # axe.legend()
    # pyplot.show(block=False)
    # c1 = show_3D(Z.T, Zobs.T)
    # c1.title = "Sans bruit"
    # c2 = show_3D(Z20.T, Zobs.T)
    # c2.title = "Bruit 1/20"
    # c3 = show_3D(Z50.T, Zobs.T)
    # c3.title = "Bruit 1/50"
    #
    # vispy.app.run()


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


def show_2D(*datas, labels=None, savepath=None):
    fig = pyplot.figure()
    axe = fig.gca()
    D = len(datas)
    cm = matplotlib.cm.rainbow(np.linspace(0, 1, D))
    for i, data in enumerate(datas):
        color = cm[i]
        label = labels[i] if labels is not None else f"Group {i}"
        axe.scatter(*data.T, c=color, label=label)
    axe.legend()
    if savepath:
        pyplot.savefig(savepath)
    pyplot.show()


def tsne():
    c = LabContextOlivine()
    Yobs1 = c.get_observations()

    c2 = LabContextNontronite()
    Yobs2 = c2.get_observations()

    _, Ytrain = c.get_data_training(10000)

    group = np.concatenate((Yobs1, Yobs2, Ytrain), axis=0)
    ti = time.time()
    tsne = TSNE(n_components=2)
    visu = tsne.fit_transform(group)
    v1, v2, vtrain = visu[:len(Yobs1)], visu[len(Yobs1) + 1:len(Yobs2) + len(Yobs1)], visu[len(Yobs2) + len(Yobs1) + 1:]
    logging.info(f"Stochastic Neighbor Embedding done in {time.time() - ti:.3f} s")
    labels = ["Olivine", "Nontronite", "Données d'entrainement"]
    # axe = show_3D(v1.T,v2.T,vtrain.T,labels=["Olivine","Nontronite","Données d'entrainement"])
    # vispy.app.run()

    show_2D(v1, v2, vtrain, labels=labels, savepath="/home/bkugler/Documents/reunion8_11/tsne.png")


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # show_diff_learning()
    tsne()
