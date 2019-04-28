import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox, \
    QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtCore, QtGui, QtWidgets


# ############################## TARGETED TRAJECTORY GENERATOR ############################## #

class Targeter(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)

        self.plot_trajectory_targeted = GraphWidgets.GraphView3D()

        self.bottom_panel = QFrame()
        self.bottom_layout = QHBoxLayout()
        self.bottom_panel.setLayout(self.bottom_layout)
        self.layout.addWidget(self.plot_trajectory_targeted)
        self.layout.addWidget(self.bottom_panel)

        self.state = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
        self.end_seconds = 10
        self.resolution = self.end_seconds / 2
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.reference_orbit = None
        self.thresh_min = 0
        self.thresh_max = 10

    def specify_trajectory(self, state, end_seconds, reference_orbit, thresh_min, thresh_max):

        self.reference_orbit = reference_orbit
        self.state = state
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.end_seconds = end_seconds
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))

        self.populate_targeted_trajectory()

    def populate_targeted_trajectory(self):

        current_time = 0
        end_times = []
        maneuver = 0
        targeted_state = [self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5]]

        trajectory_history = [[], [], [], [], [], []]
        while current_time < self.end_seconds:
            dv, results, current_time, target_status = J2RelativeMotion.j2_sedwick_propagator(targeted_state,
                                                                                              self.reference_orbit,
                                                                                              self.times,
                                                                                              self.times[1] - self.times[0],
                                                                                              3, self.thresh_min,
                                                                                              self.thresh_max, False)
            end_times.append(current_time)
            # print("Time until out-of-bounds: ", current_time)
            current_time = 0
            # let's now archive the propagation we've completed
            trajectory_history[0].append(results[0])
            trajectory_history[1].append(results[1])
            trajectory_history[2].append(results[2])
            trajectory_history[3].append(results[3])
            trajectory_history[4].append(results[4])
            trajectory_history[5].append(results[5])

            maneuver = maneuver + 1

            if target_status:
                break
            elif maneuver > 7:
                break

            # using the dv we've completed, we can now alter the initial state and try again
            targeted_state[3] = dv[3]
            targeted_state[4] = dv[4]
            targeted_state[5] = dv[5]

        self.plot_trajectory_targeted.update_graph([[trajectory_history[0][-1], trajectory_history[1][-1], trajectory_history[2][-1]], ],
                                                   "Targeted Motion for " + str(self.end_seconds) + " seconds | Trajectory: " +
                                                   str(self.state), ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])
