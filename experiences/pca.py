import matplotlib.cm
import matplotlib
import numpy as np
import vispy.scene

matplotlib.use("Qt5Agg")
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot
from vispy import scene, app, color
from vispy.scene import visuals

from sklearn.decomposition import PCA

from experiences import rtls


def show_diff_learning():
    h = rtls.RtlsCO2()
    Yobs = h.get_observations()

    _, Y = h.get_data_training(100000)

    Ynoise20 = h.add_noise_data(Y, std=20)
    Ynoise50 = h.add_noise_data(Y, std=50)

    d1 = np.mean([np.linalg.norm(Y - yobs, axis=1).min() for yobs in Yobs])
    d2 = np.mean([np.linalg.norm(Ynoise20 - yobs, axis=1).min() for yobs in Yobs])
    d3 = np.mean([np.linalg.norm(Ynoise50 - yobs, axis=1).min() for yobs in Yobs])

    print(d1, d2, d3)

    print(np.abs(Y - Ynoise20).mean())
    print(np.abs(Y - Ynoise50).mean())
    return

    pca = PCA(n_components=3, copy=True)

    pca.fit(Y)

    Z = pca.transform(Y)
    Zobs = pca.transform(Yobs)
    Z20 = pca.transform(Ynoise20)
    Z50 = pca.transform(Ynoise50)

    fig = pyplot.figure()
    fig.add_subplot(111, projection="3d")
    axe = fig.gca()
    axe.scatter(*Z.T, label="train 0")
    axe.scatter(*Zobs.T, label="obs")
    axe.scatter(*Z20.T, label="train 1/20")
    axe.scatter(*Z50.T, label="train 1/50")
    axe.legend()
    pyplot.show(block=False)
    c1 = show_3D(Z.T, Zobs.T)
    c1.title = "Sans bruit"
    c2 = show_3D(Z20.T, Zobs.T)
    c2.title = "Bruit 1/20"
    c3 = show_3D(Z50.T, Zobs.T)
    c3.title = "Bruit 1/50"

    vispy.app.run()


def show_3D(*datas):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor="white")
    view = canvas.central_widget.add_view()

    pos, colors = [], []
    D = len(datas)
    cm = matplotlib.cm.rainbow(np.linspace(0, 1, D))
    for i, data in enumerate(datas):
        pos.extend(data.T)
        colors.extend([cm[i]] * len(data.T))
    colors = np.array(colors)
    pos = np.array(pos)

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color=None, face_color=colors, size=5)

    view.add(scatter)

    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)

    return canvas


if __name__ == '__main__':
    show_diff_learning()
