import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox, \
    QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtCore, QtGui, QtWidgets


# ############################## RELATIVE TRAJECTORY GENERATOR ############################## #

class RelativeLocator(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)

        self.plot_regions = GraphWidgets.GraphView3D()
        self.plot_trajectory = GraphWidgets.GraphView3D()
        self.plot_trajectory_magnitude = GraphWidgets.GraphView2D()

        self.display_pass_times = QFrame()
        self.display_pass_times_layout = QVBoxLayout()
        self.display_pass_times_table = QTableWidget()
        self.display_pass_times_graph = GraphWidgets.GraphView2D()
        self.display_pass_times_layout.addWidget(self.display_pass_times_table)
        self.display_pass_times_layout.addWidget(self.display_pass_times_graph)
        self.display_pass_times_graph.hide()
        self.display_pass_times_switch_button = QPushButton("change display")
        self.display_pass_times_switch_button.clicked.connect(self.when_display_pass_times_switch_button_clicked)
        self.display_pass_times_layout.addWidget(self.display_pass_times_switch_button)
        self.display_pass_times.setLayout(self.display_pass_times_layout)

        # set column count
        self.display_pass_times_table.setColumnCount(2)
        self.display_pass_times_table.setVerticalHeaderLabels(["Pass Number", "Pass Duration"])
        self.display_pass_times_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.display_pass_times_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        self.layout.addWidget(self.plot_regions)
        self.layout.addWidget(self.plot_trajectory)
        self.layout.addWidget(self.display_pass_times)
        self.layout.addWidget(self.plot_trajectory_magnitude)

        self.plot_regions.hide()
        self.display_pass_times.hide()
        self.plot_trajectory_magnitude.hide()

        self.bottom_panel = QFrame()
        self.bottom_layout = QHBoxLayout()

        self.plot_regions_button = QPushButton("plot regions")
        self.plot_regions_button.clicked.connect(self.when_plot_regions_button_clicked)
        self.plot_trajectory_button = QPushButton("plot trajectory")
        self.plot_trajectory_button.clicked.connect(self.when_plot_trajectory_button_clicked)
        self.display_pass_times_button = QPushButton("show pass times")
        self.display_pass_times_button.clicked.connect(self.when_display_pass_times_button_clicked)
        self.display_magnitude_button = QPushButton("show distance magnitude")
        self.display_magnitude_button.clicked.connect(self.when_plot_magnitude_button_clicked)

        self.bottom_layout.addWidget(self.plot_regions_button)
        self.bottom_layout.addWidget(self.plot_trajectory_button)
        self.bottom_layout.addWidget(self.display_pass_times_button)
        self.bottom_layout.addWidget(self.display_magnitude_button)

        self.enter_trajectory = QLineEdit("")  # enter in format "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"

        self.bottom_panel.setLayout(self.bottom_layout)

        self.layout.addWidget(self.bottom_panel)

        self.thresh_min = 0
        self.thresh_max = 10
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_seconds = 20000
        self.resolution = self.end_seconds
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.trajectory = [[], [], []]
        self.reference_orbit = None

    def when_display_pass_times_switch_button_clicked(self):
        if self.display_pass_times_graph.isVisible():
            self.display_pass_times_graph.hide()
            self.display_pass_times_table.show()
        else:
            self.display_pass_times_table.hide()
            self.display_pass_times_graph.show()

    def when_plot_regions_button_clicked(self):
        self.display_pass_times.hide()
        self.plot_trajectory.hide()
        self.plot_trajectory_magnitude.hide()
        self.plot_regions.show()

    def when_plot_trajectory_button_clicked(self):
        self.plot_regions.hide()
        self.display_pass_times.hide()
        self.plot_trajectory_magnitude.hide()
        self.plot_trajectory.show()

    def when_display_pass_times_button_clicked(self):
        self.plot_regions.hide()
        self.plot_trajectory.hide()
        self.plot_trajectory_magnitude.hide()
        self.display_pass_times.show()

    def when_plot_magnitude_button_clicked(self):
        self.plot_regions.hide()
        self.plot_trajectory.hide()
        self.display_pass_times.hide()
        self.plot_trajectory_magnitude.show()

    def specify_trajectory(self, state, end_seconds, reference_orbit, thresh_min, thresh_max):
        self.reference_orbit = reference_orbit
        self.state = state
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.end_seconds = end_seconds
        self.resolution = self.end_seconds
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.populate_region_graph()
        self.populate_trajectory_graph()
        self.populate_time_graph()
        self.populate_magnitude_graph()

    def populate_region_graph(self):
        t_in, data_in, t_out, data_out = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                                                self.times,
                                                                                self.times[1] - self.times[0],
                                                                                2, self.thresh_min, self.thresh_max, False)
        data = [data_in, data_out]
        self.plot_regions.update_scatter(data,
                                    "Relative Motion for " + str(self.end_seconds) + " seconds | Trajectory: " + str(self.state),
                                    ["Within Range", "Out of Range"],
                                    ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])

    def populate_trajectory_graph(self):

        print(self.state)

        trajectory = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                            self.times, self.times[1] - self.times[0],
                                                            0, self.thresh_min, self.thresh_max, False)

        self.plot_trajectory.update_graph([trajectory, ],
                                  "Relative Motion for " + str(self.end_seconds) + " seconds | Trajectory: " + str(self.state),
                                  ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])

    def populate_time_graph(self):

        times = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                       self.times, self.times[1] - self.times[0],
                                                       3, self.thresh_min, self.thresh_max, False)

        # set row count
        self.display_pass_times_table.setRowCount(len(times))

        for i in range(len(times)):
            self.display_pass_times_table.setItem(i, 1, QTableWidgetItem(str(times[i])))
            self.display_pass_times_table.setItem(i, 0, QTableWidgetItem(str(i)))

        self.display_pass_times_graph.update_graph([times, ],
                                    "Opportunities for " + str(self.end_seconds)[:8] + " seconds | Trajectory: " +
                                    str(self.state[0])[:7] + ", " + str(self.state[1])[:7] + ", " +
                                    str(self.state[2])[:7] + ", " + str(self.state[3])[:7] + ", " +
                                    str(self.state[4])[:7] + ", " + str(self.state[5])[:7] + ", ",
                                    ["Pass Number", "Amount of Time (s)"])

    def populate_magnitude_graph(self):
        magnitudes = J2RelativeMotion.j2_sedwick_propagator(self.state, self.reference_orbit,
                                                       self.times, self.times[1] - self.times[0],
                                                       4, self.thresh_min, self.thresh_max, False)

        self.plot_trajectory_magnitude.update_graph([magnitudes, ],
                                    "Distance Magnitude for " + str(self.end_seconds)[:8] + " seconds | Trajectory: " +
                                    str(self.state[0])[:7] + ", " + str(self.state[1])[:7] + ", " +
                                    str(self.state[2])[:7] + ", " + str(self.state[3])[:7] + ", " +
                                    str(self.state[4])[:7] + ", " + str(self.state[5])[:7] + ", ",
                                    ["Time (s)", "Distance (m)"])

