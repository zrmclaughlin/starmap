import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import StarMap
import CombinedRelativeMotion
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox
from PyQt5.QtCore import pyqtSlot


# ############################## HEAT MAP GENERATOR ############################## #

class HeatMap(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)
        self.heatmap_plot = GraphWidgets.GraphViewHeatMap()
        self.layout.addWidget(self.heatmap_plot)

        self.maximum_time_x_axis_values = []
        self.maximum_time_y_axis_values = []
        self.maximum_times = []

        self.num_axis_points = 10
        self.x_axis = 0
        self.x_units = " N/A "
        self.x_property = "Unfilled"

        self.num_axis_points = 10
        self.y_axis = 1
        self.y_units = " N/A "
        self.y_property = "Unfilled"

        self.current_trajectory = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
        self.end_seconds = 10
        self.reference_orbit = None

        self.bottom_panel = QFrame()
        self.bottom_layout = QHBoxLayout()

        self.valuable_trajectory_dropdown = QComboBox()
        self.valuable_trajectory_list = []
        self.valuable_trajectory_dropdown.activated.connect(self.choose_trajectory_to_propagate)

        self.bottom_layout.addWidget(self.valuable_trajectory_dropdown)
        self.bottom_panel.setLayout(self.bottom_layout)

        self.layout.addWidget(self.bottom_panel)

    def choose_trajectory_to_propagate(self, text):
        print(text)
        self.current_trajectory[self.x_axis] = self.maximum_time_x_axis_values[text]
        self.current_trajectory[self.y_axis] = self.maximum_time_y_axis_values[text]

    def heat_map_xy(self, x_variance, y_variance, mean_state,
                    reference_orbit, end_seconds, recorded_times,
                    x_axis, y_axis, threshold_min, threshold_max):

        self.end_seconds = end_seconds
        self.current_trajectory = mean_state
        self.reference_orbit = reference_orbit

        self.x_axis = x_axis
        if (self.x_axis == 0) | (self.x_axis == 1) | (self.x_axis == 2):
            self.x_units = " m "
            if self.x_axis == 0:
                self.x_property = " Radial "
            elif self.x_axis == 1:
                self.x_property = " In-Track "
            else:
                self.x_property = " Cross-Track "
        else:
            self.x_units = " m/s "
            if self.x_axis == 3:
                self.x_property = " Radial "
            elif self.x_axis == 4:
                self.x_property = " In-Track "
            else:
                self.x_property = " Cross-Track "

        self.y_axis = y_axis
        if (self.y_axis == 0) | (self.y_axis == 1) | (self.y_axis == 2):
            self.y_units = " m "
            if self.y_axis == 0:
                self.y_property = " Radial: "
            elif self.y_axis == 1:
                self.y_property = " In-Track: "
            else:
                self.y_property = " Cross-Track: "
        else:
            self.y_units = " m/s "
            if self.y_axis == 3:
                self.y_property = " Radial: "
            elif self.y_axis == 4:
                self.y_property = " In-Track: "
            else:
                self.y_property = " Cross-Track: "

        x_values_list = np.linspace(-x_variance+mean_state[self.x_axis],
                                    x_variance+mean_state[self.x_axis],
                                    self.num_axis_points).tolist()
        y_values_list = np.linspace(-y_variance+mean_state[self.y_axis],
                                    y_variance+mean_state[self.y_axis],
                                    self.num_axis_points).tolist()

        success_level = np.zeros([self.num_axis_points, self.num_axis_points])
        times = np.linspace(0.0, self.end_seconds, recorded_times)

        step = times[1] - times[0]

        for i in range(self.num_axis_points):

            for j in range(self.num_axis_points):

                mean_state[self.x_axis] = x_values_list[i]
                mean_state[self.y_axis] = y_values_list[j]

                if StarMap.GLOBAL_BACKEND == "COMBINED":
                    success_level[i][j] = CombinedRelativeMotion.j2_drag_ecc_propagator(mean_state, reference_orbit, times, step,
                                                                                 1, threshold_min, threshold_max, False)\
                                                                                 / recorded_times * 100
                elif StarMap.GLOBAL_BACKEND == "J2":
                    success_level[i][j] = J2RelativeMotion.j2_sedwick_propagator(mean_state, reference_orbit, times, step,
                                                                                 1, threshold_min, threshold_max, False)\
                                                                                 / recorded_times * 100
                else:
                    success_level[i][j] = J2RelativeMotion.j2_sedwick_propagator(mean_state, reference_orbit, times, step,
                                                                                 1, threshold_min, threshold_max, False)\
                                                                                 / recorded_times * 100

        map_x, map_y = np.array(np.meshgrid(x_values_list, y_values_list))
        max_x, max_y = np.where(np.asarray(success_level) == np.asarray(success_level).max())

        axis_labels = [self.x_property + "Variance" + self.x_units,
                       self.y_property + " Variance" + self.y_units,
                       "% of Time within Constraint"]
        title = "Relative Motion Heatmap for " + str(self.end_seconds) + " seconds"

        self.heatmap_plot.update_graph(map_x, map_y, success_level, title, axis_labels)

        self.valuable_trajectory_list = []

        for i in range(len(max_x)):
            self.maximum_time_x_axis_values.append(x_values_list[max_x[i]])
            self.maximum_time_y_axis_values.append(y_values_list[max_y[i]])
            self.maximum_times.append(success_level[max_x[i]][max_y[i]])
            self.valuable_trajectory_list.append(self.x_property + str(x_values_list[max_x[i]]) + self.x_units + " | " +
                                                 self.y_property + str(y_values_list[max_y[i]]) + self.y_units + " | " +
                                                 " Time: " + str(success_level[max_x[i]][max_y[i]]))

        self.current_trajectory = mean_state

        self.valuable_trajectory_dropdown.clear()
        self.valuable_trajectory_dropdown.addItems(self.valuable_trajectory_list)



# ################################################################################ #