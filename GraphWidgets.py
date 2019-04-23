from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget,QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from itertools import cycle
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
plt.style.use("seaborn")
matplotlib.rcParams.update({'font.size': 8})


class GraphView2D(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.dpi = 100
        self.fig = Figure((5.0, 3.0), dpi=self.dpi, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.layout.setStretchFactor(self.canvas, 1)
        self.setLayout(self.layout)
        self.font = QtGui.QFont()
        self.font.setPointSize(1)
        self.canvas.show()

    def update_graph(self, data, title, labels, axis_labels):
        self.axes.clear()
        colors = cycle(
            ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
        for item in range(len(data)):
            self.axes.scatter(data[item], color=next(colors), label=labels[item])
        self.axes.legend(loc="best")
        plt.setp(self.axes.get_xticklabels(), rotation=70, horizontalalignment='right', fontsize=5)
        self.axes.set_title(title)
        self.axes.set_xlabel(axis_labels[0])
        self.axes.set_ylabel(axis_labels[1])
        self.canvas.draw()


class GraphViewHeatMap(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.dpi = 100
        self.fig = Figure((5.0, 3.0), dpi=self.dpi, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.layout.setStretchFactor(self.canvas, 1)
        self.setLayout(self.layout)
        self.font = QtGui.QFont()
        self.font.setPointSize(1)
        self.canvas.show()

    def update_graph(self, map_x, map_y, data, title, axis_labels):
        self.axes.clear()
        surf = self.axes.plot_surface(map_x, map_y, data, cmap=cm.YlGnBu,
                               linewidth=0, antialiased=False)
        self.axes.set_title(title)
        self.axes.set_xlabel(axis_labels[0])
        self.axes.set_ylabel(axis_labels[1])
        self.axes.set_zlabel(axis_labels[2])
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        self.axes.mouse_init()
        self.canvas.draw()
