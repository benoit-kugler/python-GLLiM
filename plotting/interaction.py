import vispy
from matplotlib.animation import FuncAnimation

vispy.use(app="PyQt5")

import vispy.app
from matplotlib import pyplot, gridspec, axes
from vispy.plot import Fig

class AxesSequence():
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self,is_3D=False):
        self.fig = pyplot.figure()
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

        self.is_3D = is_3D

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        if self.is_3D:
            ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
                                   visible=False, label=self._n,projection="3d")
        else:
            ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes) - 1:
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show_first(self):
        self.axes[0].set_visible(True)



class SubplotsSequence():

    def __init__(self,nb_row,nb_column,nb_subplot,figsize=None):
        self.axesS = []  # iter, plots

        self.fig = pyplot.figure(figsize=figsize)

        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.nb_row = nb_row
        self.nb_column = nb_column
        self.nb_subplot = nb_subplot


    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        axes = []
        spec = gridspec.GridSpec(ncols=self.nb_column,nrows=self.nb_row)
        for i in range(self.nb_subplot):
            if i == self.nb_subplot - 1:
                ind = i - (self.nb_row-1)*self.nb_column
                ax = self.fig.add_subplot(spec[self.nb_row - 1,ind : ],
                                          visible=False, label=str(self._n) + "_" + str(i))
            else:
                ax = self.fig.add_subplot(self.nb_row,self.nb_column,i+1,
                                   visible=False,label=str(self._n) + "_" + str(i) )
            axes.append(ax)

        self._n += 1
        self.axesS.append(axes)
        return axes

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axesS) - 1:
            for axe in self.axesS[self._i]:
                axe.set_visible(False)
            for axe in self.axesS[self._i+1]:
                axe.set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            for axe in self.axesS[self._i]:
                axe.set_visible(False)
            for axe in self.axesS[self._i+-1]:
                axe.set_visible(True)
            self._i -= 1

    def show_first(self):
        for axe in self.axesS[0]:
            axe.set_visible(True)


class vispyAnimation():

    AXE_TITLE = "Updating axe"
    """Graph title"""

    INTERVAL = 1
    """Time in s between 2 frames"""

    def __init__(self,iterations):
        self.fig = Fig()
        self.fig.events.key_release.connect(self.on_key_press)
        self.init_axe()
        self.timer = vispy.app.Timer(interval=self.INTERVAL,connect=self.update,
                                start=False,iterations=iterations)

        self.fig.show(run=True)

    def on_key_press(self,event):
        if event.key == "Space":
            self.resume()
        elif event.key == "Left":
            self.reset()

    def resume(self):
        if self.timer.running:
            self.timer.stop()
        elif self.timer.iter_count == self.timer.max_iterations:
            print("Sequence already completed !")
        else:
            self.timer.start()

    def reset(self):
        self.current_frame = 0
        if self.timer.running:
            self.timer.stop()
            self.timer.start(iterations=self.timer.max_iterations)
        else:
            self.timer = vispy.app.Timer(interval=self.INTERVAL,connect=self.update,
                                start=False,iterations=self.timer.max_iterations)


    def init_axe(self):
        self.current_frame = 0
        self.axe = self.fig[0,0]

    def update(self,event):
        self._draw()
        self.current_frame += 1

    def _draw(self):
        pass


class mplAnimation():
    INTERVAL = 200

    axe: axes.Axes
    fig: pyplot.Figure

    def __init__(self, data, xlabel="x1", ylabel=None, xlims=(0, 1), ylims=None):

        self.fig, self.axe = pyplot.subplots()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlims = xlims
        self.ylims = ylims

        self.ani = FuncAnimation(self.fig, self.update, frames=enumerate(data),
                                 init_func=self.init_animation, blit=True,
                                 interval=self.INTERVAL)
        self.is_paused = False
        self.fig.canvas.mpl_connect("button_press_event", self.onPause)
        pyplot.show()

    def init_animation(self):
        self.axe.set_xlim(*self.xlims)
        if self.ylims is not None:
            self.axe.set_ylim(*self.ylims)
        self.axe.set_xlabel("$" + self.xlabel + "$")
        if self.ylabel:
            self.axe.set_ylabel("$" + self.ylabel + "$")
        return []

    def update(self, frame):
        pass

    def onPause(self, event):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()

if __name__ == '__main__':
    vispyAnimation(1)