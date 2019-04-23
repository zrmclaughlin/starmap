import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import OrbitalElements
import HeatMap
import GraphWidgets


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'STARMAP v0.1'
        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 400
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = StarMapGUI(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class StarMapGUI(QWidget):
 
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.ic_tab =  QWidget()  # initial conditions tab
        self.heatmap_tab = HeatMap.HeatMap()
        self.tabs.resize(600, 400)

        self.x_pos = QLineEdit("0.0")
        self.x_p_var = QLineEdit("0.1")
        self.y_pos = QLineEdit("0.0")
        self.y_p_var = QLineEdit("0.1")
        self.z_pos = QLineEdit("0.01")
        self.z_p_var = QLineEdit("0.1")
        self.x_vel = QLineEdit("0.0")
        self.x_v_var = QLineEdit("0.1")
        self.y_vel = QLineEdit("0.0")
        self.y_v_var = QLineEdit("0.1")
        self.z_vel = QLineEdit("0.01")
        self.z_v_var = QLineEdit("0.1")

        self.state_elements = [self.x_pos, self.y_pos, self.z_pos, self.x_vel, self.y_vel, self.z_vel]
        self.state_variances = [self.x_p_var, self.y_p_var, self.z_p_var, self.x_v_var, self.y_v_var, self.z_v_var]

        self.reference_inclination = QLineEdit(".52")
        self.reference_semimajor_axis = QLineEdit("6678136.6")
        self.maximum_distance_threshold = QLineEdit("30")
        self.minimum_distance_threshold = QLineEdit("0")

        self.propagation_time = QLineEdit("20000")
        self.values_record = QLineEdit("1000")

        self.x_resolution = QLineEdit("3")
        self.y_resolution = QLineEdit("3")
        self.heatmap_drop_down_menu_x_axis = QComboBox(self)
        self.heatmap_drop_down_menu_y_axis = QComboBox(self)

        self.heatmap_x_axis = 3
        self.heatmap_y_axis = 4

        # Add tabs
        self.tabs.addTab(self.ic_tab, "Initial Conditions")
        self.tabs.addTab(self.heatmap_tab, "XY Velocity HeatMap")

        self.startButton = QPushButton("Get Simulation From Entered Conditions")
        self.startButton.clicked.connect(self.when_startButton_clicked)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Create textboxes
        self.relmot_frame = QFrame()
        self.reforbit_frame = QFrame()
        self.resolution_frame = QFrame()
        self.timing_frame = QFrame()
        self.threshold_frame = QFrame()

        self.set_ic_tab_relmot_layout()
        self.set_ic_tab_reforbit_layout()
        self.set_ic_resolution_layout()
        self.set_ic_timing_layout()
        self.set_ic_tab_threshold_layout()

        total_layout = QVBoxLayout()
        total_layout.addWidget(self.reforbit_frame)
        total_layout.addWidget(self.threshold_frame)
        total_layout.addWidget(self.resolution_frame)
        total_layout.addWidget(self.timing_frame)
        total_layout.addWidget(self.relmot_frame)
        total_layout.addWidget(self.startButton)

        self.ic_tab.setLayout(total_layout)

    def set_ic_tab_reforbit_layout(self):
        ic_layout = QHBoxLayout()

        label_row = QLabel()
        label_row.setText("Ref. inclination")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.reference_inclination)

        label_row = QLabel()
        label_row.setText("Ref. semi-major axis")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.reference_semimajor_axis)

        self.reforbit_frame.setLayout(ic_layout)


    def set_ic_tab_threshold_layout(self):
        ic_layout = QHBoxLayout()

        label_row = QLabel()
        label_row.setText("Max. position threshold")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.maximum_distance_threshold)

        label_row = QLabel()
        label_row.setText("Min. position threshold")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.minimum_distance_threshold)

        self.threshold_frame.setLayout(ic_layout)

    def set_ic_timing_layout(self):
        ic_layout = QHBoxLayout()

        label_row = QLabel()
        label_row.setText("Time for propagation")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.propagation_time)

        label_row = QLabel()
        label_row.setText("# Values to record")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.values_record)

        item_list = ["X Velocity", "X Position", "Y Velocity", "Y Position", "Z Velocity", "Z Position"]
        self.heatmap_drop_down_menu_x_axis.activated.connect(self.heatmap_choice_x)
        self.heatmap_drop_down_menu_x_axis.addItems(item_list)
        ic_layout.addWidget(self.heatmap_drop_down_menu_x_axis)

        item_list = ["Y Velocity", "Y Position", "X Velocity", "X Position", "Z Velocity", "Z Position"]
        self.heatmap_drop_down_menu_y_axis.activated.connect(self.heatmap_choice_y)
        self.heatmap_drop_down_menu_y_axis.addItems(item_list)
        ic_layout.addWidget(self.heatmap_drop_down_menu_y_axis)

        self.resolution_frame.setLayout(ic_layout)

    def heatmap_choice_x(self, text):
        if text == 0:
            self.heatmap_x_axis = 3
        elif text == 1:
            self.heatmap_x_axis = 0
        if text == 2:
            self.heatmap_x_axis = 4
        elif text == 3:
            self.heatmap_x_axis = 1
        if text == 4:
            self.heatmap_x_axis = 5
        elif text == 5:
            self.heatmap_x_axis = 2

    def heatmap_choice_y(self, text):
        if text == 0:
            self.heatmap_y_axis = 4
        elif text == 1:
            self.heatmap_y_axis = 1
        if text == 2:
            self.heatmap_y_axis = 3
        elif text == 3:
            self.heatmap_y_axis = 0
        if text == 4:
            self.heatmap_y_axis = 5
        elif text == 5:
            self.heatmap_y_axis = 2

    def set_ic_resolution_layout(self):
        ic_layout = QHBoxLayout()

        label_row = QLabel()
        label_row.setText("X-Axis resolution")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.x_resolution)

        label_row = QLabel()
        label_row.setText("Y-Axis resolution")
        ic_layout.addWidget(label_row)
        ic_layout.addWidget(self.y_resolution)

        self.timing_frame.setLayout(ic_layout)

    def set_ic_tab_relmot_layout(self):
        ic_layout_titles = ["Radial Position", "In-Track Position", "Cross-Track Position",
                            "Radial Velocity", "In-Track Velocity", "Cross-Track Velocity"]
        condition_number = 6
        ic_layout = QGridLayout()

        ic_layout.addWidget(QLabel("Parameter"), 0, 0)
        ic_layout.addWidget(QLabel("Mean Value"), 0, 1)
        ic_layout.addWidget(QLabel("+/- Variance"), 0, 2)

        for index in range(condition_number):
            label_row = QLabel()
            label_row.setText(ic_layout_titles[index])

            ic_layout.addWidget(label_row, index+1, 0)
            ic_layout.addWidget(self.state_elements[index], index+1, 1)
            ic_layout.addWidget(self.state_variances[index], index+1, 2)

        self.relmot_frame.setLayout(ic_layout)

    @pyqtSlot()
    def when_startButton_clicked(self):

        reference_inclination = float(self.reference_inclination.text())
        reference_semimajor_axis = float(self.reference_semimajor_axis.text())
        maximum_distance_threshold = float(self.maximum_distance_threshold.text())

        x_pos = float(self.x_pos.text())
        x_p_var = float(self.x_p_var.text())
        y_pos = float(self.y_pos.text())
        y_p_var = float(self.y_p_var.text())
        z_pos = float(self.z_pos.text())
        z_p_var = float(self.z_p_var.text())
        x_vel = float(self.x_vel.text())
        x_v_var = float(self.x_v_var.text())
        y_vel = float(self.y_vel.text())
        y_v_var = float(self.y_v_var.text())
        z_vel = float(self.z_vel.text())
        z_v_var = float(self.z_v_var.text())

        mean_state = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
        reference_orbit = OrbitalElements.OrbitalElements(reference_semimajor_axis, 0.0001,
                                                          reference_inclination, 0.0, 0.0, 0.0, 3.986004415E14)

        end_seconds = 20000
        recorded_times = 1000

        print(self.heatmap_x_axis, self.heatmap_y_axis)

        # xy_v_heatmap = GraphWidgets.GraphViewHeatMap()
        # HeatMap.heat_map_xy(xy_v_heatmap, x_v_var, y_v_var, mean_state, 30, 30, reference_orbit, end_seconds,
        #                     recorded_times, self.heatmap_x_axis, self.heatmap_y_axis, 0, maximum_distance_threshold)
        self.heatmap_tab.x_axis_points = int(self.x_resolution.text())
        self.heatmap_tab.y_axis_points = int(self.y_resolution.text())
        self.heatmap_tab.heat_map_xy(x_v_var, y_v_var, mean_state, reference_orbit, end_seconds,
                            recorded_times, self.heatmap_x_axis, self.heatmap_y_axis, 0, maximum_distance_threshold)
        # xy_v_heatmap_layout = QHBoxLayout()
        # xy_v_heatmap_layout.addWidget(xy_v_heatmap)

        # self.xy_v_heatmap_tab.setLayout(xy_v_heatmap_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    INIT_STARMAP = App()
    sys.exit(app.exec_())
