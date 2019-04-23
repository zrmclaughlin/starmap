import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox


# ############################## HEAT MAP GENERATOR ############################## #

class HeatMap(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.heatmap_plot = GraphWidgets.GraphViewHeatMap()
        self.maximum_time_x_axis_values = []
        self.maximum_time_y_axis_values = []
        self.maximum_times = []
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.heatmap_plot)
        self.x_axis_points = 10
        self.y_axis_points = 10
        self.x_axis = 0
        self.y_axis = 1
        self.current_trajectory = [0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
        self.valuable_trajectory_dropdown = QComboBox()
        self.valuable_trajectory_list = []
        self.valuable_trajectory_dropdown.activated.connect(self.choose_trajectory_to_propagate)
        self.layout.addWidget(self.valuable_trajectory_dropdown)

    def choose_trajectory_to_propagate(self, text):
        self.current_trajectory[self.x_axis] = self.maximum_time_x_axis_values[text]
        self.current_trajectory[self.y_axis] = self.maximum_time_y_axis_locations[text]

    def heat_map_xy(self, x_variance, y_variance, mean_state,
                    reference_orbit, end_seconds, recorded_times,
                    x_axis, y_axis, threshold_min, threshold_max):

        self.current_trajectory = mean_state

        self.x_axis = x_axis
        self.y_axis = y_axis

        x_values_list = np.linspace(-x_variance, x_variance, self.x_axis_points).tolist()
        y_values_list = np.linspace(-y_variance, y_variance, self.y_axis_points).tolist()

        success_level = np.zeros([self.x_axis_points, self.y_axis_points])
        times = np.linspace(0.0, end_seconds, recorded_times)

        step = times[1] - times[0]

        for i in range(self.x_axis_points):

            for j in range(self.y_axis_points):

                mean_state[self.x_axis] = x_values_list[i]
                mean_state[self.y_axis] = y_values_list[j]

                success_level[i][j] = J2RelativeMotion.j2_sedwick_propagator(mean_state, reference_orbit, times, step,
                                                                             False, threshold_min, threshold_max)\
                                                                             / recorded_times * 100

        map_x, map_y = np.array(np.meshgrid(x_values_list, y_values_list))
        max_x, max_y = np.where(np.asarray(success_level) == np.asarray(success_level).max())
        print(max_x, max_y)

        axis_labels = ["Radial Variance (m/s)", "In-Track Variance (m/s)", "Percentage of Time within Constraint"]
        title = "Relative Motion Heatmap for " + str(end_seconds) + " seconds"

        self.heatmap_plot.update_graph(map_x, map_y, success_level, title, axis_labels)

        for i in range(len(max_x)):
            self.maximum_time_x_axis_values.append(x_values_list[max_x[i]])
            self.maximum_time_y_axis_values.append(y_values_list[max_y[i]])
            self.maximum_times.append(success_level[max_x[i]][max_y[i]])
            self.valuable_trajectory_list.append("Radial: " + str(x_values_list[max_x[i]]) + "units" + " In-Track: " +
                                                 str(y_values_list[max_y[i]]) + "units" +
                                                 " Time: " + str(success_level[max_x[i]][max_y[i]]))

        self.valuable_trajectory_dropdown.clear()
        self.valuable_trajectory_dropdown.addItems(self.valuable_trajectory_list)


# ################################################################################ #