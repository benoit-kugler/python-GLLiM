import PIL
import matplotlib
import numpy as np
from cartopy import crs
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")

from matplotlib import cm, ticker, transforms, rc, axes
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from plotting.interaction import AxesSequence, SubplotsSequence, vispyAnimation, mplAnimation


def _latex_relative_error(est,true):
    return r"Relative error : $\frac{{|| {est} - {true}||_{{2}} }} {{||{true}||_{{2}} }}$".\
        format(est=est,true=true)

def _get_rows_columns(N,coeff_row = 5, coeff_column=5):
    """Helper to position abritary number of subplots"""
    nb_row = np.ceil(np.sqrt(N))
    nb_column = np.ceil(N / nb_row)
    figsize= (nb_column * coeff_column,nb_row * coeff_row)
    return nb_row, nb_column, figsize


def overlap_colors(rnk, with_base=False):
    """Mixes base colours according to cluster importance (given by rnk).
    rnk shape : (N,K)"""
    _, K = rnk.shape
    base_colours = cm.rainbow(np.arange(K) / K)
    rn = rnk.sum(axis=1)
    c = rnk.dot(base_colours) / rn[:, None]
    c[:, 3] = (rn <= 1) * rn + (rn > 1) * 1
    if with_base:
        return c, base_colours
    return c


class abstractDrawerMPL():
    FIGURE_TITLE_BOX = dict(boxstyle="round", facecolor='#D8D8D8',
                            ec="0.5", pad=0.5, alpha=1)

    Y_TITLE_BOX_WITH_CONTEXT = 1.2
    Y_TITLE_BOX_WITHOUT_CONTEXT = 1.05
    fig: Figure

    def __init__(self, *args, title="", savepath=None, context=None, draw_context=False, write_context=False):
        self.create_figure(*args)
        print("Drawing...")

        self.set_title(title, context, draw_context)
        self.main_draw(*args)


        if write_context:
            self.write_context(context, savepath)

        if savepath:
            self.save(savepath)

    def create_figure(self, *args):
        self.fig = pyplot.figure()

    def main_draw(self, *args):
        pass

    def set_title(self, title, context, draw_context):
        context = self._format_context(context) if context is not None else ""
        if draw_context:
            title = title + "\n" + context
        y = self.Y_TITLE_BOX_WITH_CONTEXT if draw_context else self.Y_TITLE_BOX_WITHOUT_CONTEXT
        self.fig.suptitle(title, bbox=self.FIGURE_TITLE_BOX, y=y)

    @staticmethod
    def _format_context(context):
        """Retuns a matplolib compatible string which describes metadata."""
        context["with_noise"] = "-" if context["with_noise"] is None else context["with_noise"]
        context["added"] = "-" if context["added"] is None else context["added"]
        context["para"] = context["para"] or "No"
        s = """
        Data $\\rightarrow N_{{train}}={N}$ (+ {Nadd}) ; $N_{{test}}={Ntest}$ ; Noise : {with_noise} ; 
                Partial : {partiel} ;  Method : {method} ; Added training : {added}. 
        Estimator $\\rightarrow$ Class : {gllim_class} ; Second learning : {para}.
        Constraints $\\rightarrow$ $\Sigma$ : {sigma_type} ; $\Gamma$  : {gamma_type}. 
        Mixture $\\rightarrow$ $K={K}$ ; $L_{{w}}$={Lw} ; Init with local cluster : {init_local}"""
        return s.format(**context)

    @classmethod
    def write_context(cls, context, savepath):
        context = cls._format_context(context)
        with open(savepath[:-4] + ".tex", "w") as f:
            f.write(context)

    def save(self, savepath):
        self.fig.savefig(savepath, bbox_inches='tight', pad_inches=0.2)
        print("Saved in", savepath)
        pyplot.close(self.fig)


class clusters(abstractDrawerMPL):

    def main_draw(self, X, rnk, ck, varnames, xlims):
        colors, base_colors = overlap_colors(rnk, with_base=True)
        varx, vary = varnames
        xlim, ylim = xlims
        axe = self.fig.gca()
        axe.scatter(*X.T, c=colors, alpha=0.5, marker=".")
        axe.scatter(*ck.T, s=50, c=base_colors, marker="o")
        axe.set_xlim(*xlim)
        axe.set_ylim(*ylim)
        axe.set_xlabel(r'${}$'.format(varx))
        axe.set_ylabel(r'${}$'.format(vary))


class simple_plot(abstractDrawerMPL):

    def main_draw(self, values, labels, xlabels, ylog):
        rc("text", usetex=True)
        xlabels = xlabels or list(range(len(values[0])))
        axe = self.fig.gca()
        for v, lab, m in zip(values, labels, [".", "+", "+", "+", "+", "+", "o", "o", "o", ",", ",", ",", "."]):
            axe.scatter(xlabels, v, marker=m, label=lab)
        if ylog:
            axe.set_yscale("log")
        axe.set_ylim(np.array(values).min(), np.array(values).max())
        axe.tick_params(axis="x", labelsize=7)
        axe.legend()

    def save(self, savepath):
        super().save(savepath)
        rc("text", usetex=False)


class plusieursKN(simple_plot):

    @staticmethod
    def _format_context(context):
        return f"""Courbe 1: $K$ évolue linéairement et $N = {context['coeffNK']} K$. 
                Courbe 2: idem pour $K$ mais $N = {context['coeffmaxN1']}  K_{{max}}$.
                Courbe 3: item pour $K$ mais $N = {context['coeffmaxN2']}  K_{{max}}$."""


def _axe_schema_1D_direct(axe, ck, ckS, Ak, bk, xlims):
    s = axe.scatter(ck, ckS, marker="+", color='r', label="$(c_{k}, c_{k}^{*})$")

    x_box = np.linspace(0, (xlims[1] - xlims[0]) / 50, 100)
    artists = [s]
    for k, a, b in zip(range(len(bk)), Ak[:, 0, 0], bk[:, 0]):
        x = x_box + ck[k]
        y = a * x + b
        p = axe.plot(x, y, color="g", alpha=0.7)
        artists.extend(p)
    return artists

class schema_1D(abstractDrawerMPL):
    Y_TITLE_BOX_WITH_CONTEXT = 1.1

    @staticmethod
    def _format_context(context):
        p = context["init_local"]
        if p is None:
            return f"Initialisation usuelle. Après apprentissage, $\max\limits_{{k}} \Gamma_{{k}}$ = {context['max_Gamma']:.1e}"
        else:
            return f"Initialisation avec précision = {p}. Après apprentissage, $\max\limits_{{k}} \Gamma_{{k}}$ = {context['max_Gamma']:.1e}"

    def main_draw(self, points_true_F, ck, ckS, Ak, bk, xlims, xtrue, ytest, modal_preds):
        axe = self.fig.add_subplot(2, 1, 1)
        axe.set_xlim(*xlims)

        axe.plot(*points_true_F, color="b", label="True F")
        _axe_schema_1D_direct(axe, ck, ckS, Ak, bk, xlims)

        if ytest is not None:
            axe = self.fig.add_subplot(2, 1, 2)
            axe.scatter(ck, ckS, marker="+", color='r', label="$(c_{k}, c_{k}^{*})$")
            axe.plot(*points_true_F, color="b", label="True F")

            axe.axhline(y=ytest)
            for i, (xpred, w) in enumerate(modal_preds):
                axe.axvline(xpred[0], color="y", linewidth=0.5, label="{0:.4f} - {1:.2f}".format(xpred[0], w))
                axe.annotate(str(i), (xpred[0], 0))
            if xtrue is not None:
                axe.axvline(xtrue, linestyle="--", color="g", alpha=0.5, label="$x_{initial}$")
        self.fig.legend(*axe.get_legend_handles_labels(), bbox_to_anchor=(1.01, 0.8))
        self.fig.subplots_adjust(right=0.77)


### --------------- Plot de plusieurs subplots sur une grille ----------------- ###

class abstractGridDrawerMPL(abstractDrawerMPL):
    """Plots severals graphs on the same figure, with shared context."""

    SIZE_ROW = 5
    SIZE_COLUMN = 5

    AXES_3D = False

    def _get_nb_subplot(self, *args):
        return 1

    def create_figure(self, *args):
        self.nb_row, self.nb_column, figsize = _get_rows_columns(self._get_nb_subplot(*args),
                                                                 coeff_row=self.SIZE_ROW, coeff_column=self.SIZE_COLUMN)
        self.fig = pyplot.figure(figsize=figsize)

    def save(self, savepath):
        self.fig.tight_layout(rect=[0.2, 0, 1, 1])
        super().save(savepath)

    def get_axes(self):
        n = 1
        projection = "3d" if self.AXES_3D else None
        while True:
            axe = self.fig.add_subplot(self.nb_row, self.nb_column, n, projection=projection)
            n += 1
            yield axe


class estimated_F(abstractGridDrawerMPL):
    AXES_3D = True

    def _get_nb_subplot(self, X, Y, Y_components, data_trueF, rnk, varnames, varlims):
        return len(Y_components)

    def main_draw(self, X, Y, Y_components, data_trueF, rnk, varnames, varlims):
        colors = "b" if rnk is None else overlap_colors(rnk)

        varx, vary = varnames
        xlim, ylim = varlims

        x, y, zs_true = data_trueF
        for g, axe in zip(Y_components, self.get_axes()):
            axe.set_title("Component {}".format(g))
            axe.scatter(*X.T, Y[:, g], c=colors, marker=".", s=1, alpha=0.6)
            z = zs_true[g]
            axe.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.7, label="True F")
            axe.set_xlim(*xlim)
            axe.set_ylim(*ylim)
            axe.set_xlabel(r'${}$'.format(varx))
            axe.set_ylabel(r'${}$'.format(vary))


class plot_density2D(abstractGridDrawerMPL):
    RESOLUTION = 200
    SIZE_COLUMN = 5
    SIZE_ROW = 3

    def _get_nb_subplot(self, fs, varlims, varnames, titles, modal_preds, trueXs, colorplot,
                        var_description):
        return len(fs)

    def main_draw(self, fs, varlims, varnames, titles, modal_preds, trueXs, colorplot,
                  var_description):

        for f, (xlim, ylim), modal_pred, truex, varname, title, axe in zip(fs, varlims,
                                                                           modal_preds, trueXs, varnames, titles,
                                                                           self.get_axes()):
            x, y = np.meshgrid(np.linspace(*xlim, self.RESOLUTION, dtype=float),
                               np.linspace(*ylim, self.RESOLUTION, dtype=float))
            variable = np.array([x.flatten(), y.flatten()]).T
            print("Comuting of density...")
            z, _ = f(variable)
            print("Done.")
            z = z.reshape((self.RESOLUTION, self.RESOLUTION))
            _axe_density2D(self.fig, axe, x, y, z, colorplot, xlim, ylim, varname,
                           modal_pred, truex, title)

        if len(fs) > 0:
            handles, labels = axe.get_legend_handles_labels()
            self.fig.legend(handles, labels, loc="center left")

        self.fig.text(0.5, -0.1, var_description, horizontalalignment='center',
                      fontsize=12, bbox=self.FIGURE_TITLE_BOX, fontweight='bold')


### ----------- Sequence interactive ------------- ###

class abstractSequence(abstractDrawerMPL):
    """Switchable sequence of plots"""

    def create_figure(self, *args):
        self.axes = AxesSequence()
        self.fig = self.axes.fig()


class clusters_one_by_one(abstractSequence):

    def main_draw(self, X, rnk, ck, varnames, varlims):
        varx, vary = varnames
        xlim, ylim = varlims
        _, K = rnk.shape
        colors = cm.rainbow(np.arange(K) / K)
        for k, base_c, axe, cc in zip(range(K), colors, self.axes, ck):
            c = [(*base_c[0:3], p) for p in rnk[:, k]]
            axe.scatter(*X.T, label=str(k), c=c)
            axe.scatter(*cc, s=50, color=base_c, marker="+", label=f"c_{ {k} }")
            axe.legend()
            axe.set_xlim(*xlim)
            axe.set_ylim(*ylim)
            axe.set_xlabel(r'${}$'.format(varx))
            axe.set_ylabel(r'${}$'.format(vary))

        self.axes.show_first()
        self.fig.show()


### ------------ Histograms --------------- ###

class abstractHistogram(abstractDrawerMPL):
    TITLE = "Histogram"
    XLABEL = "x"
    LABELS = None

    def set_title(self, title, context, draw_context):
        super().set_title(self.TITLE, context, draw_context)

    def main_draw(self, values, cut_tail, labels):
        xlabel = self.XLABEL
        if cut_tail:
            xlabel += " - Cut tail : {}%".format(cut_tail)
            values = [sorted(error)[:-len(error) * cut_tail // 100] for error in values]

        error_max = max(sum(values, []))
        bins = np.linspace(0, error_max, 1000)
        axe = self.fig.gca()
        m = len(values)
        alphas = [0.5 + i / (2 * m) for i in range(m)]
        labels = labels or self.LABELS
        for serie, alpha, label in zip(values, alphas, labels):
            axe.hist(serie, bins, alpha=alpha, label=label)
        axe.legend()
        axe.set_ylabel("Test points number")
        axe.set_xlabel(xlabel)

        means = [np.mean(s) for s in values]
        medians = [np.median(s) for s in values]

        stats = ["{0} $\\rightarrow$ Mean : {1:.2E} ; Median : {2:.2E}".format(label, mean, median)
                 for mean, median, label in zip(means, medians, labels)]
        s = "\n".join(stats)

        self.fig.text(0.5, -0.07 * m, s, horizontalalignment='center',
                      fontsize=10, bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                                             ec="0.5", pad=0.5, alpha=1), fontweight='bold')


class hist_Flearned(abstractHistogram):
    TITLE = "Comparaison beetween $F$ and it's estimation"
    XLABEL = _latex_relative_error("F_{est}(x)", "F(x)")
    LABELS = ["Mean estimation"]


class hist_retrouveYmean(abstractHistogram):
    TITLE = """Comparaison beetween $Y_{obs}$ and $F(x_{pred})$ 
         for mean prediction"""
    XLABEL = _latex_relative_error("F(x_{pred})", "Y")
    LABELS = ["Cohérence (prédiction par la moyenne)"]


class hist_retrouveY(abstractHistogram):
    TITLE = """Comparaison beetween $Y_{obs}$ and $F(x_{pred})$ 
         (several $x_{pred}$ are found for each $y$)"""
    XLABEL = _latex_relative_error("F(x_{pred})", "Y")


class hist_retrouveYbest(abstractHistogram):
    TITLE = """Comparaison beetween $Y_{obs}$ and $F(x_{pred})$ 
         (best $x_{pred}$ for each $y$)"""
    XLABEL = _latex_relative_error("F(x_{pred})", "Y")


class hist_modalPrediction(abstractHistogram):
    TITLE = "Comparaison beetween X and the best modal prediction"
    XLABEL = _latex_relative_error("X_{best}", "X")


class hist_meanPrediction(abstractHistogram):
    TITLE = "Comparaison beetween X and it's mean prediction"
    XLABEL = _latex_relative_error("X_{est}", "X")


##### -------------- Animation -------------------- ####

class EvolutionCluster2D(vispyAnimation):
    INTERVAL = 0.05
    AXE_TITLE = "Clusters evolution"

    def __init__(self, points, rnks, density, xlim, ylim):
        self.points = points
        self.rnks = rnks[:, :, :100]
        self.density = density
        self.xlim = xlim
        self.ylim = ylim
        imax, _, self.K = self.rnks.shape
        super().__init__(imax)

    def init_axe(self):
        super().init_axe()
        self.line = self.axe.plot(self.points, width=0, symbol="disc", marker_size=2, edge_width=0)
        self.axe.title.text = "X clusters"
        self.axe2 = self.fig[0, 1]
        self.axe2.title.text = "X density"
        self.line2 = self.axe2.plot(self.points, width=0, symbol="disc", marker_size=2, edge_width=0)
        self._draw()

    def reset(self):
        super().reset()
        self._draw()

    def _draw(self):
        self.fig.title = "Iteration {}".format(self.current_frame)
        rnk = self.rnks[self.current_frame]
        c = overlap_colors(rnk)
        self.line._markers.set_data(pos=self.points, face_color=c, edge_color=c)
        dens = self.density[self.current_frame]
        c2 = cm.coolwarm(dens / dens.max())
        self.line2._markers.set_data(pos=self.points, face_color=c2, edge_color=c2)


class Evolution1D(mplAnimation):

    def __init__(self, points_true_F, ck, ckS, Ak, bk, xlims):
        self.points_true_F = points_true_F
        data = list(zip(ck, ckS, Ak, bk))
        super().__init__(data,xlabel="x",xlims=xlims)


    def init_animation(self):
        super().init_animation()
        xF, yF = self.points_true_F
        s = self.axe.scatter(xF, yF, marker=".", color="b", label="True F", s=0.3)
        return s,

    def update(self, frame):
        i, (ck, ckS, Ak, bk) = frame
        artists = _axe_schema_1D_direct(self.axe, ck, ckS, Ak, bk, self.xlims)
        l = self.axe.legend([f"Iteration {i}"], loc="lower left")
        # t = self.axe.text(0.01,0.01,f"Iteration {i}")
        return artists + [l]


##### ----------------- TODO A refactor en classe  --------------- #################






def _axe_density_1D(axe,x,y,xlims,
                   varnames,modal_preds,truex,title):

    # axe.xaxis.set_minor_locator(ticker.MultipleLocator((xlims[1] - xlims[0]) / 100))
    axe.xaxis.set_major_locator(ticker.MultipleLocator((xlims[1] - xlims[0])/20))
    axe.plot(x,y,"-",linewidth=1)
    axe.set_xlim(*xlims)
    axe.set_xlabel("$" + varnames[0] + "$")
    axe.set_title(title)

    # for i,yi in enumerate(sub_densities):
    #     axe.plot(x_points,yi.flatten(),"--",label="Dominant {}".format(i),linewidth=0.5)

    for i, (X, height, weight) in enumerate(modal_preds):
        axe.axvline(x=X, color="b",linestyle="--",
                    label="X modal {0:.2e} - {1:.3f}".format(height,weight),alpha=0.5)

    if truex:
        axe.axvline(x=truex,label="True value",alpha=0.5)




def plot_density1D(fs,contexte,xlims=((0,1),),resolution=200,main_title="Density 1D",titles=("",),
                   modal_preds=((),),trueXs=None,
                   var_description = "",filename=None,varnames=(("x",),)):

    trueXs = trueXs or [None] * len(fs)
    nb_row, nb_column, figsize = _get_rows_columns(len(fs),coeff_column=5,coeff_row=4)
    fig = pyplot.figure(figsize=figsize)
    n = 1

    for f , xlim, modal_pred, truex, varname, title in zip(fs,xlims,modal_preds,
                                                           trueXs,varnames, titles):
        x = np.linspace(*xlim,resolution)[:,None]
        y , _ = f(x)
        axe = fig.add_subplot(nb_row,nb_column,n)

        _axe_density_1D(axe,x.flatten(),y.flatten(),xlim,
                   varname,modal_pred,truex,title)
        n += 1

    fig.text(0.5, 0, var_description, horizontalalignment='center',
             fontsize=12, bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                                    ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    if fs:
        handles, labels = axe.get_legend_handles_labels()
        fig.legend(handles,labels,bbox_to_anchor=(0.18,1))

    title = main_title + "\n" + contexte
    fig.suptitle(title, bbox=FIGURE_TITLE_BOX,y = 1.18)
    fig.tight_layout(rect=[0.2,0,1,1])
    if filename:
        fig.savefig(filename, bbox_inches='tight')
        print("Saved in ", filename)
        pyplot.close(fig)

def _axe_density2D(fig,axe,x,y,z,colorplot,xlims,ylims,
                   varnames,modal_preds,truex,title):
    if colorplot:
        pc = axe.pcolormesh(x,y,z)
        axe.set_xlim(*xlims)
        axe.set_ylim(*ylims)
        fig.colorbar(pc)
    else:
        levels = np.linspace(0,z.max(),20)
        levels = [0.001] + list(levels)
        axe.contour(x,y,z,levels=levels,alpha=0.5)
    axe.set_xlabel("$"+ varnames[0]  + "$")
    axe.set_ylabel("$"+ varnames[1]  + "$")

    colors = cm.coolwarm(np.arange(len(modal_preds)) / len(modal_preds))
    for i, (X,height,weight) in enumerate(modal_preds):
        axe.scatter(X[0],X[1], color=colors[i],marker=".",label="X modal {0:.2e} - {1:.3f}".format(height,weight))
        axe.annotate(str(i),(X[0],X[1]))

    if truex is not None:
        axe.scatter(truex[0], truex[1], color="r", marker="+", label="True value",s = 50,zorder=10)

    axe.set_title(title)









def trace_retrouveY(Xs,diffs):
    """Xs shape (N,nb_component,L) diffs shape (N,nb_component,D)"""
    D = diffs.shape[2]
    for i in range(Xs.shape[2]): # One variable for each figure
        fig = pyplot.figure()
        axe = fig.add_subplot(111, projection='3d')
        xs ,ys,zs = [],[],[]
        for xsn , ysn in zip(Xs,diffs):
            for x,y in zip(xsn,ysn):
                v = x[i]
                xs.extend([v] * D )
                ys.extend( list(range(D)))
                zs.extend(y)
        axe.scatter(xs,ys,zs)
    pyplot.show()








def influence_theta(B0,H):
    thetas = np.linspace(0,np.pi,200)
    b = B0 * H / ( H + np.tan(thetas/2))
    pyplot.plot(thetas,1+b)
    pyplot.xlabel(r"$\theta$ en radian")
    pyplot.ylabel(r"$1 + B$")
    pyplot.show()


def show_projections(X,labels=None,varnames=("x1","x2","x3","x4")):
    """Plot 4 first dims of X, 2 per 2"""
    f = pyplot.figure()
    n = 1
    if labels is not None:
        colors = cm.coolwarm(np.arange(max(labels) + 1 ) / max(labels) )
        colors =  [colors[c] for c in labels]
    else:
        colors = None
    for i in range(4):
        for j in range(i+1,4):
            a = f.add_subplot(3,2,n)
            n += 1
            x = X[:,i]
            y = X[:,j]
            a.scatter(x,y,color=colors)
            a.set_xlabel("$" + varnames[i] + "$")
            a.set_ylabel("$" +varnames[j] + "$")
    pyplot.show()


def plot_Y(Y):
    """Plot each Y with a different color (geomtries in x axis)"""
    fig = pyplot.figure()
    axe = fig.gca()
    colors = cm.rainbow(np.arange(len(Y))/len(Y))
    for y,c in zip(Y,colors):
        axe.plot(y,c=c)
    pyplot.show()


def correlations2D(X,labels_value,contexte,varnames,varlims,main_title="Corrélations"
                   ,savepath=None,add_points=None):
    nb_var = len(varnames)
    nb_row,nb_col,figsize = _get_rows_columns(nb_var*(nb_var+1)/2)
    fig = pyplot.figure(figsize=figsize)
    n= 1
    for i in range(nb_var):
        for j in range(i+1,nb_var):
            x = X[:,(i,j)]
            axe = fig.add_subplot(nb_row,nb_col,n)
            n += 1
            l = axe.scatter(*x.T,c=labels_value,marker="+")
            axe.set_xlim(varlims[i])
            axe.set_ylim(varlims[j])
            axe.set_xlabel("$" + varnames[i] + "$")
            axe.set_ylabel("$" + varnames[j] + "$")
            if add_points and (i,j) in add_points:
                x,y = add_points[(i,j)]
                axe.scatter(x,y,marker=".",color="g",alpha=0.5,s=0.7)

    title = main_title + "\n" + contexte
    fig.suptitle(title, bbox=FIGURE_TITLE_BOX,y = 1.1)
    fig.tight_layout(rect=[0,0,1,1])
    fig.colorbar(l,orientation="horizontal")
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print("Saved in ", savepath)
        pyplot.close(fig)

def correlations1D(Xmean,Xweight,StdMean,labels_value,contexte,varnames,varlims,main_title="Synthèse",savepath=None):
    nb_var = len(varnames)
    nb_row,nb_col,figsize = _get_rows_columns(nb_var,coeff_row=4,coeff_column=6)
    fig = pyplot.figure(figsize=figsize)
    for i in range(nb_var):
        axe = fig.add_subplot(nb_row * nb_col,1,i+1)
        _prediction_1D(axe,varlims[i],varnames[i],labels_value,Xmean[:,i],Xweight[:,:,i],"wavelength (microns)",
                       StdMean=StdMean[:,i,i])
    fig.legend(*axe.get_legend_handles_labels())  # pour ne pas surcharger


    title = main_title + "\n" + contexte
    fig.suptitle(title, bbox=FIGURE_TITLE_BOX,y = 1.25)
    fig.tight_layout(rect=[0,0,1,1])
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print("Saved in ", savepath)
        pyplot.close(fig)


def _prediction_1D(axe,xlim,varname,xlabels,Xmean,Xweight,xtitle,
                   StdMean=None,Yref=None,StdRef=None):
    if xlim is not None:
        axe.set_ylim(*xlim)
        axe.yaxis.set_major_locator(ticker.MultipleLocator((xlim[1] - xlim[0]) / 10))

    axe.set_xlabel(xtitle)
    axe.set_ylabel("$" + varname + "$")

    axe.plot(xlabels, Xmean, marker="*", label="Mean")
    if StdMean is not None:
        axe.fill_between(xlabels, Xmean - StdMean, Xmean + StdMean, alpha=0.3,
                          color="gray", label="Std on mean", hatch="/")
    axe.plot(xlabels, Xweight[:, 0], marker="+", label="Weight 1")
    axe.plot(xlabels, Xweight[:, 1], marker="+", label="Weight 2")
    axe.plot(xlabels, Xweight[:, 2], marker="+", label="Weight 3")

    if Yref is not None:
        axe.plot(xlabels, Yref, marker=".", label="Reference")
        axe.fill_between(xlabels, Yref + StdRef, Yref - StdRef, alpha=0.3, color="g", label="Std on reference")


def density_sequences1D(fs, modal_preds, xlabels, Xmean, Xweight, Xheight, xlim=(0, 1), title="Density sequence",
                        varname="x", resolution=200, Yref=None, StdRef=None, StdMean=None,
                        images_paths=None,savepath=None):
    if images_paths:
        axes_seq = SubplotsSequence(2,2,3,figsize=(25,15))
    else:
        axes_seq = SubplotsSequence(2,1,2,figsize=(25,15))

    x = np.linspace(*xlim, resolution)[:, None]

    for i , axes, f, m in zip(range(len(fs)),axes_seq,fs,modal_preds):
        y, _ = f(x)
        xpoints = x.flatten()
        _axe_density_1D(axes[0],xpoints,y.flatten(),xlim,varname,m,None,"")


        if images_paths:
            axe2 = axes[2]
            axe_im = axes[1]
            img =PIL.Image.open(images_paths[i]).convert("L")
            axe_im.imshow(np.asarray(img),cmap="gray", aspect="auto")
            axe_im.get_xaxis().set_visible(False)
            axe_im.get_yaxis().set_visible(False)
        else:
            axe2 = axes[1]

        _prediction_1D(axe2,xlim,varname,xlabels,Xmean,Xweight,"wavelength (microns)",
                       StdMean = StdMean, Yref=Yref, StdRef=StdRef)

        #Current point
        axe2.axvline(xlabels[i], c="b", marker="<", label="index " + str(i),zorder=4,alpha=0.4)

        axe2.legend()

    axes_seq.fig.suptitle(title)
    axes_seq.show_first()
    if savepath:
        axes_seq.fig.savefig(savepath, bbox_inches=transforms.Bbox.from_bounds(2,1,22,6))
        print("Saved in ", savepath)
    pyplot.show()


class CkAnimation(mplAnimation):

    def __init__(self,cks,varnames=("x1","x2"), varlims=((0,1),(0,1))):
        super().__init__(cks,xlabel=varnames[0],ylabel=varnames[1],
                         xlims=varlims[0],ylims=varlims[1])


    def init_animation(self):
        self.ln, = self.axe.plot([], [], 'ro', animated=True)
        return self.ln ,

    def update(self,frame):
        self.ln.set_data(frame.T)
        return self.ln,



def map_values(latlong,values,addvalues=None,main_title="Map",titles=None,savepath=None):
    fig = pyplot.figure(figsize=(25,10))
    print(latlong)
    lat = list(latlong[:,0])
    long = list(latlong[:,1])
    ax = fig.add_subplot(121,projection=crs.PlateCarree())

    if addvalues is None:
        s = ax.scatter(long, lat, transform=crs.Geodetic(),c=values,cmap=cm.rainbow)
        fig.colorbar(s)
    else:
        totalvalues = list(values) + list(addvalues)
        vmin, vmax = min(totalvalues),max(totalvalues)
        ax.scatter(long, lat , transform=crs.Geodetic(),
                         cmap=cm.rainbow, vmin=vmin,vmax=vmax,
                         c=list(values))
        ax2 = fig.add_subplot(122,projection=crs.PlateCarree())
        s2 = ax2.scatter(long, lat , transform=crs.Geodetic(),cmap=cm.rainbow,vmin=vmin,vmax=vmax,
                        c=addvalues )

        ax.set_title(titles[0])
        ax2.set_title(titles[1])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(s2,cax=cbar_ax)



    fig.suptitle(main_title)
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print("Saved in ", savepath)
        pyplot.close(fig)
    else:
        pyplot.show()


def illustre_derivative(F,dF):
    resolution = 100
    x, y = np.meshgrid(np.linspace(0,1, resolution, dtype=float),
                       np.linspace(0,30, resolution, dtype=float))
    X = np.array([x.flatten(), y.flatten()]).T
    z = F(X)
    fig = pyplot.figure()
    axe = fig.add_subplot(111,projection='3d')
    axe.plot_surface(x,y,z[:,0].reshape((resolution,resolution)),label="F")
    x0 = np.array([0.8,20])
    A = dF(x0)
    Y = A.dot((X - x0).T).T + F(x0[None,:])[0]
    axe.plot_surface(x,y,Y[:,0].reshape((resolution,resolution)),label="dF")

    pyplot.show()


if __name__ == '__main__':
    # latlong= np.array([[-75,43],[-10,0],[20,43] ])[:,(1,0)]
    # map_values(latlong,[0,1,2])
    aS = SubplotsSequence(2,2,3)
    for i , l in zip(range(10),aS):
        for a in l:
            a.plot([i] * 10)
    aS.show_first()
    pyplot.show()


