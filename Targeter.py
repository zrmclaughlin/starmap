import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox, \
    QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtCore, QtGui, QtWidgets
import TargetingUtils


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
        self.resolution = self.end_seconds
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))
        self.reference_orbit = None
        self.thresh_min = 0
        self.thresh_max = 10

        self.targeted_state = []
        self.state_transition = []
        self.desired_state = []

    def specify_trajectory(self, state, desired_state, end_seconds, reference_orbit, thresh_min, thresh_max):

        self.reference_orbit = reference_orbit
        self.state = state
        self.desired_state = desired_state
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.end_seconds = end_seconds
        self.resolution = self.end_seconds
        self.times = np.linspace(0.0, self.end_seconds, int(self.resolution))

        self.populate_targeted_trajectory()

    def populate_targeted_trajectory(self):

        targeted_state = [  # Initial relative state
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5],
            # State Transition Matrix (I, 6x6)
            # .2, 0, 0.1, .1, 0, .5,
            1.0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0,
            0, 0, 1.0, 0, 0, 0,
            0, 0, 0, 1.0, 0, 0,
            0, 0, 0, 0, 1.0, 0,
            0, 0, 0, 0, 0, 1.0]

        current_time = 0
        end_times = []
        maneuver = 0
        step = self.times[1] - self.times[0]
        trajectory_history = [[], [], [], [], [], []]
        targeted_state_history = []
        while current_time < self.end_seconds:
            dv, results, current_time, target_status = J2RelativeMotion.j2_sedwick_targeter(targeted_state, self.desired_state,
                                                                                              self.reference_orbit,
                                                                                              self.times,
                                                                                              step, self.end_seconds,
                                                                                              self.thresh_min,
                                                                                              self.thresh_max, False)
            end_times.append(current_time)
            targeted_state_history.append(targeted_state)
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
            elif maneuver > 10:
                break

            # using the dv we've completed, we can now alter the initial state and try again
            targeted_state[3] = dv[3]
            targeted_state[4] = dv[4]
            targeted_state[5] = dv[5]

        closest_time = np.amax(end_times)
        closest_time_loc = np.where(np.asarray(end_times) == np.amax(np.asarray(end_times)))

        best_run = closest_time_loc[0][0]
        self.targeted_state = [targeted_state_history[best_run][0], targeted_state_history[best_run][1], targeted_state_history[best_run][2],
                               targeted_state_history[best_run][3], targeted_state_history[best_run][4], targeted_state_history[best_run][5]]

        self.state_transition = TargetingUtils.recompose(targeted_state_history[best_run])

        self.plot_trajectory_targeted.update_graph([[trajectory_history[0][best_run], trajectory_history[1][best_run], trajectory_history[2][best_run]], ],
                                                   "Targeted Motion for " + str(closest_time)[:8] + " seconds | Trajectory: " +
                                                   str(self.targeted_state[0])[:7] + ", " + str(self.targeted_state[1])[:7] + ", " +
                                                   str(self.targeted_state[2])[:7] + ", " + str(self.targeted_state[3])[:7] + ", " +
                                                   str(self.targeted_state[4])[:7] + ", " + str(self.targeted_state[5])[:7] + ", ",
                                                   ["Radial (m)", "In-Track (m)", "Cross-Track (m)"])
