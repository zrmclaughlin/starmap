import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame, QComboBox
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtGui
import OrbitalElements
import HeatMap
import RelativeLocator
import GraphWidgets
import random
import Targeter


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
        # Initial Conditions Tab
        self.ic_tab = QWidget()
        # HeatMap Tab
        self.heatmap_tab = HeatMap.HeatMap()
        self.select_heatmap_trajectory_button = QPushButton("select")
        self.heatmap_tab.bottom_layout.addWidget(self.select_heatmap_trajectory_button)
        self.select_heatmap_trajectory_button.clicked.connect(self.when_heatmap_button_clicked)

        self.target_tab = Targeter.Targeter()
        self.select_targeted_trajectory_button = QPushButton("select")
        self.target_tab.bottom_layout.addWidget(self.select_targeted_trajectory_button)
        self.select_targeted_trajectory_button.clicked.connect(self.when_targeted_trajectory_button_clicked)

        # Relative Location Tab
        self.relloc_tab = RelativeLocator.RelativeLocator()
        # Size tabs
        self.tabs.resize(600, 400)

        self.x_pos = QLineEdit("0.1")
        self.x_p_var = QLineEdit("0.1")
        self.y_pos = QLineEdit("0.1")
        self.y_p_var = QLineEdit("0.1")
        self.z_pos = QLineEdit("0.01")
        self.z_p_var = QLineEdit("0.1")
        self.x_vel = QLineEdit("-0.02")
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

        self.propagation_time = QLineEdit("10000")
        self.values_record = QLineEdit("1000")

        self.resolution = QLineEdit("3")
        self.heatmap_drop_down_menu_x_axis = QComboBox(self)
        self.heatmap_drop_down_menu_y_axis = QComboBox(self)

        self.heatmap_x_axis = 3
        self.heatmap_y_axis = 4
        self.maximum_distance_threshold_value = None
        self.minimum_distance_threshold_value = None
        self.reference_orbit = None

        # Add tabs
        self.tabs.addTab(self.ic_tab, "Initial Conditions")
        self.tabs.addTab(self.heatmap_tab, "Initial State HeatMap")
        self.tabs.addTab(self.target_tab, "Targeted Trajectory")
        self.tabs.addTab(self.relloc_tab, "Relative Trajectory")

        self.start_button_heatmap = QPushButton("Get HeatMap From Entered Conditions")
        self.start_button_heatmap.clicked.connect(self.when_start_button_heatmap_clicked)

        self.select_relloc_trajectory_button = QPushButton("Get Trajectory From Entered Conditions")
        self.select_relloc_trajectory_button.clicked.connect(self.when_start_relloc_button_clicked)

        self.select_target_trajectory_button = QPushButton("Target a Trajectory From Entered Conditions")
        self.select_target_trajectory_button.clicked.connect(self.when_start_button_target_clicked)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Create textboxes
        self.starmap_name_frame = QFrame()
        self.relmot_frame = QFrame()
        self.reforbit_frame = QFrame()

        self.set_ic_title_layout()
        self.set_ic_tab_orbit_threshold_timing()

        total_layout = QVBoxLayout()
        total_layout.addWidget(self.starmap_name_frame)
        total_layout.addWidget(self.reforbit_frame)
        total_layout.addWidget(self.start_button_heatmap)
        total_layout.addWidget(self.select_relloc_trajectory_button)
        total_layout.addWidget(self.select_target_trajectory_button)

        self.ic_tab.setLayout(total_layout)

    def get_app_title_message(self):
        title_string = ['<b> Welcome to STARMAP! </b>', '<b> im gonna FREAK IT </b>', '<b> first... i park my car </b>',
                        '<b> im going FERAL </b>', '<b> me when I get you </b>', '<b> ;) </b>',
                        '<b> im feeling the effect... </b>', '<b> the sensation </b>',
                        '<b> what if we kissed in walmart and we were both heterosexual :O </b>', '<b> hi im evil </b>',
                        '<b> slug academy </b>', '<b> going :airplane_emoji: buffalo milk </b>',
                        '<b> the anti effect </b>', '<b> Time to plot this plant... </b>', '<b> Understood. </b>',
                        '<b> you DONT have the super monkey ball controller :/ </b>',
                        '<b> you DONT have the apple tv remote </b>']
        return random.choice(title_string)

    def set_ic_title_layout(self):
        ic_layout = QHBoxLayout()
        font_setting = QtGui.QFont()
        font_setting.setPointSize(40)
        gui_title = QLabel(self.get_app_title_message())
        gui_title.setFont(font_setting)
        gui_title.setAlignment(Qt.AlignCenter)
        ic_layout.addWidget(gui_title)
        self.starmap_name_frame.setLayout(ic_layout)

    def set_ic_tab_orbit_threshold_timing(self):

        ic_layout = QGridLayout()

        initial_state_subtitle_1 = QLabel("<b>Reference</b>")

        ic_layout.addWidget(initial_state_subtitle_1, 0, 2)
        ic_layout.addWidget(QLabel("<b>Properties</b>"), 0, 3)

        ic_layout.addWidget(QLabel("Ref. inclination"), 1, 0)
        ic_layout.addWidget(self.reference_inclination, 1, 1)
        ic_layout.addWidget(QLabel("radians"), 1, 2)

        ic_layout.addWidget(QLabel("Ref. semi-major axis"), 1, 3)
        ic_layout.addWidget(self.reference_semimajor_axis, 1, 4)
        ic_layout.addWidget(QLabel("meters"), 1, 5)

        ic_layout.addWidget(QLabel("Max. position"), 2, 0)
        ic_layout.addWidget(self.maximum_distance_threshold, 2, 1)
        ic_layout.addWidget(QLabel("meters"), 2, 2)

        ic_layout.addWidget(QLabel("Min. position"), 2, 3)
        ic_layout.addWidget(self.minimum_distance_threshold, 2, 4)
        ic_layout.addWidget(QLabel("meters"), 2, 5)

        ic_layout.addWidget(QLabel("Propagation time"), 3, 0)
        ic_layout.addWidget(self.propagation_time, 3, 1)
        ic_layout.addWidget(QLabel("seconds"), 3, 2)

        ic_layout.addWidget(QLabel("Values to record"), 3, 3)
        ic_layout.addWidget(self.values_record, 3, 4)
        ic_layout.addWidget(QLabel("#"), 3, 5)

        initial_state_subtitle_2 = QLabel("<b>Initial State</b>")

        ic_layout.addWidget(initial_state_subtitle_2, 4, 2)
        ic_layout.addWidget(QLabel("<b>Properties</b>"), 4, 3)

        ic_layout_titles = ["Radial Position", "In-Track Position", "Cross-Track Position",
                            "Radial Velocity", "In-Track Velocity", "Cross-Track Velocity"]

        ic_layout_units = ["meters", "meters", "meters",
                            "meters/second", "meters/second", "meters/second"]

        condition_number = 6

        for index in range(condition_number):
            label_row = QLabel()
            label_row.setText(ic_layout_titles[index])

            ic_layout.addWidget(label_row, index+6, 0)
            ic_layout.addWidget(self.state_elements[index], index+6, 1)
            ic_layout.addWidget(QLabel(ic_layout_units[index]), index + 6, 2)
            ic_layout.addWidget(QLabel("         +/-"), index + 6, 3)
            ic_layout.addWidget(self.state_variances[index], index+6, 4)
            ic_layout.addWidget(QLabel(ic_layout_units[index]), index + 6, 5)

        initial_state_subtitle_3 = QLabel("<b>Heat Map</b>")

        ic_layout.addWidget(initial_state_subtitle_3, 12, 2)
        ic_layout.addWidget(QLabel("<b>Properties</b>"), 12, 3)

        label_row = QLabel()
        label_row.setText("Heat Map Axes")
        ic_layout.addWidget(label_row, 13, 0)

        item_list = ["X Velocity", "X Position", "Y Velocity", "Y Position", "Z Velocity", "Z Position"]
        self.heatmap_drop_down_menu_x_axis.activated.connect(self.heatmap_choice_x)
        self.heatmap_drop_down_menu_x_axis.addItems(item_list)
        ic_layout.addWidget(self.heatmap_drop_down_menu_x_axis, 13, 1)

        item_list = ["Y Velocity", "Y Position", "X Velocity", "X Position", "Z Velocity", "Z Position"]
        self.heatmap_drop_down_menu_y_axis.activated.connect(self.heatmap_choice_y)
        self.heatmap_drop_down_menu_y_axis.addItems(item_list)
        ic_layout.addWidget(self.heatmap_drop_down_menu_y_axis, 13, 4)

        label_row = QLabel()
        label_row.setText("Heat Map Resolution")
        ic_layout.addWidget(label_row, 14, 0)
        ic_layout.addWidget(self.resolution, 14, 1)

        self.reforbit_frame.setLayout(ic_layout)

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

    def get_initial_info(self):

        reference_inclination = float(self.reference_inclination.text())
        reference_semimajor_axis = float(self.reference_semimajor_axis.text())
        self.maximum_distance_threshold_value = float(self.maximum_distance_threshold.text())
        self.minimum_distance_threshold_value = float(self.minimum_distance_threshold.text())

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
        variances = [x_p_var, y_p_var, z_p_var, x_v_var, y_v_var, z_v_var]

        self.reference_orbit = OrbitalElements.OrbitalElements(reference_semimajor_axis, 0.0001,
                                                               reference_inclination, 0.0, 0.0, 0.0,
                                                               3.986004415E14)

        end_seconds = int(self.propagation_time.text())
        recorded_times = int(self.values_record.text())
        self.heatmap_tab.num_axis_points = int(self.resolution.text())
        return mean_state, variances, end_seconds, recorded_times

    @pyqtSlot()
    def when_start_relloc_button_clicked(self):
        mean_state, variances, end_seconds, recorded_times = self.get_initial_info()
        self.relloc_tab.specify_trajectory(mean_state, end_seconds, self.reference_orbit,
                                                        self.minimum_distance_threshold_value,
                                                        self.maximum_distance_threshold_value)

    @pyqtSlot()
    def when_start_button_heatmap_clicked(self):
        mean_state, variances, end_seconds, recorded_times = self.get_initial_info()
        self.heatmap_tab.heat_map_xy(variances[self.heatmap_x_axis], variances[self.heatmap_y_axis], mean_state,
                                     self.reference_orbit, end_seconds,
                                     recorded_times, self.heatmap_x_axis, self.heatmap_y_axis,
                                     self.minimum_distance_threshold_value, self.maximum_distance_threshold_value)

    @pyqtSlot()
    def when_start_button_target_clicked(self):
        mean_state, variances, end_seconds, recorded_times = self.get_initial_info()
        self.target_tab.specify_trajectory(mean_state, end_seconds, self.reference_orbit,
                                           self.minimum_distance_threshold_value, self.maximum_distance_threshold_value)

    @pyqtSlot()
    def when_heatmap_button_clicked(self):
        self.relloc_tab.specify_trajectory(self.heatmap_tab.current_trajectory,
                                           self.heatmap_tab.end_seconds, self.reference_orbit,
                                           self.minimum_distance_threshold_value,
                                           self.maximum_distance_threshold_value)

    @pyqtSlot()
    def when_targeted_trajectory_button_clicked(self):
        self.get_initial_info()
        self.relloc_tab.specify_trajectory(self.target_tab.targeted_state,
                                           self.target_tab.end_seconds, self.reference_orbit,
                                           self.minimum_distance_threshold_value,
                                           self.maximum_distance_threshold_value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    INIT_STARMAP = App()
    sys.exit(app.exec_())
