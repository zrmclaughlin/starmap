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

        self.plot_regions = GraphWidgets.GraphView3D()
        self.plot_trajectory = GraphWidgets.GraphView3D()
        self.plot_pass_times = GraphWidgets.GraphView2D()

        self.layout.addWidget(self.plot_regions)
        self.layout.addWidget(self.plot_trajectory)
        self.layout.addWidget(self.plot_pass_times)

        self.plot_regions.hide()
        self.plot_pass_times.hide()

        self.bottom_panel = QFrame()
        self.bottom_layout = QHBoxLayout()

        self.plot_regions_button = QPushButton("plot regions")
        self.plot_regions_button.clicked.connect(self.when_plot_regions_button_clicked)
        self.plot_trajectory_button = QPushButton("plot trajectory")
        self.plot_trajectory_button.clicked.connect(self.when_plot_trajectory_button_clicked)
        self.plot_pass_times_button = QPushButton("plot pass_times")
        self.plot_pass_times_button.clicked.connect(self.when_plot_pass_times_button_clicked)

        self.bottom_layout.addWidget(self.plot_regions_button)
        self.bottom_layout.addWidget(self.plot_trajectory_button)
        self.bottom_layout.addWidget(self.plot_pass_times_button)

        self.bottom_panel.setLayout(self.bottom_layout)

        self.layout.addWidget(self.bottom_panel)

        self.thresh_min = 0
        self.thresh_max = 10
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_seconds = 20000
        self.resolution = self.end_seconds / 2
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.trajectory = [[], [], []]
        self.reference_orbit = None

    def when_plot_regions_button_clicked(self):
        self.plot_pass_times.hide()
        self.plot_trajectory.hide()
        self.plot_regions.show()

    def when_plot_trajectory_button_clicked(self):
        self.plot_regions.hide()
        self.plot_pass_times.hide()
        self.plot_trajectory.show()

    def when_plot_pass_times_button_clicked(self):
        self.plot_regions.hide()
        self.plot_trajectory.hide()
        self.plot_pass_times.show()

    def specify_trajectory(self, state, end_seconds, reference_orbit, thresh_min, thresh_max):
        self.reference_orbit = reference_orbit
        self.state = state
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.end_seconds = end_seconds

        self.populate_region_graph()
        self.populate_trajectory_graph()
        self.populate_time_graph()

    def populate_region_graph(self):
        t_in, data_in, t_out, data_out = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                                                self.times,
                                                                                self.times[1] - self.times[0],
                                                                                2, self.thresh_min, self.thresh_max)
        data = [data_in, data_out]
        self.plot_regions.update_scatter(data,
                                    "Relative Motion for " + str(self.end_seconds) + " seconds | Trajectory: " + str(self.state),
                                    ["Within Range", "Out of Range"],
                                    ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])

    def populate_trajectory_graph(self):

        trajectory = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                            self.times, self.times[1] - self.times[0],
                                                            0, self.thresh_min, self.thresh_max)

        self.plot_trajectory.update_graph([trajectory, ],
                                  "Relative Motion for " + str(self.end_seconds) + " seconds | Trajectory: " + str(self.state),
                                  ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])

    def populate_time_graph(self):

        times = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                       self.times, self.times[1] - self.times[0],
                                                       3, self.thresh_min, self.thresh_max)

        self.plot_pass_times.update_graph([times, ],
                                  "Opportunities for " + str(self.end_seconds) + " seconds | Trajectory: " + str(self.state),
                                  [""],
                                  ["Pass Number", "Amount of Time (s)"])
