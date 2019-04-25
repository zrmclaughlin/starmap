import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox
from PyQt5.QtCore import pyqtSlot


# ############################## HEAT MAP GENERATOR ############################## #

class RelativeLocator(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)
        self.plot_3d = GraphWidgets.GraphView3D()
        self.layout.addWidget(self.plot_3d)

        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_seconds = 20000
        self.resolution = self.end_seconds / 2
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.trajectory = [[], [], []]
        self.reference_orbit = None

    def specify_trajectory(self, state, end_seconds, reference_orbit, thresh_min, thresh_max):
        self.reference_orbit = reference_orbit
        self.state = state
        print(state)
        self.end_seconds = end_seconds
        # self.state = [0.0, 0.0, 0.01, 0.001, 0.0, 0.01]
        # trajectory = J2RelativeMotion.j2_sedwick_propagator(self.state, reference_orbit,
        #                                                          self.times, self.times[1] - self.times[0],
        #                                                          0, thresh_min, thresh_max)
        t_in, data_in, t_out, data_out = J2RelativeMotion.j2_sedwick_propagator(self.state, reference_orbit,
                                                                 self.times, self.times[1] - self.times[0],
                                                                 2, thresh_min, thresh_max)
        data = [data_in, data_out]
        self.plot_3d.update_scatter(data,
                                  "Relative Motion for " + str(self.end_seconds) + " seconds",
                                  ["Within Range", "Out of Range"],
                                  ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])
        # self.plot_3d.update_graph([trajectory, ],
        #                           "Relative Motion for " + str(self.end_seconds) + " seconds",
        #                           [""],
        #                           ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])
