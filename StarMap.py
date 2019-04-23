import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, \
    QTabWidget, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel, QGridLayout, QFrame
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
        self.ic_tab = QWidget()  # initial conditions tab
        self.xy_p_heatmap_tab = QWidget()
        self.xy_v_heatmap_tab = QWidget()
        self.xy_p_heatmap_tab = QWidget()
        self.tabs.resize(600, 400)

        self.x_pos = QLineEdit("0.0")
        self.x_p_var = QLineEdit("0.1")
        self.y_pos = QLineEdit("0.0")
        self.y_p_var = QLineEdit("0.1")
        self.z_pos = QLineEdit("0.001")
        self.z_p_var = QLineEdit("0.1")
        self.x_vel = QLineEdit("0.0")
        self.x_v_var = QLineEdit("0.1")
        self.y_vel = QLineEdit("0.0")
        self.y_v_var = QLineEdit("0.1")
        self.z_vel = QLineEdit("0.001")
        self.z_v_var = QLineEdit("0.1")

        self.state_elements = [self.x_pos, self.y_pos, self.z_pos, self.x_vel, self.y_vel, self.z_vel]
        self.state_variances = [self.x_p_var, self.y_p_var, self.z_p_var, self.x_v_var, self.y_v_var, self.z_v_var]

        self.reference_inclination = QLineEdit(".52")
        self.reference_semimajor_axis = QLineEdit("6678136.6")
        self.maximum_distance_threshold = QLineEdit("30")

        # Add tabs
        self.tabs.addTab(self.ic_tab, "Initial Conditions")
        self.tabs.addTab(self.xy_v_heatmap_tab, "XY Velocity HeatMap")

        self.startButton = QPushButton("Get Simulation From Entered Conditions")
        self.startButton.clicked.connect(self.when_startButton_clicked)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Create textboxes
        self.relmot_frame = QFrame()
        self.reforbit_frame = QFrame()

        self.set_ic_tab_relmot_layout()
        self.set_ic_tab_reforbit_layout()

        total_layout = QVBoxLayout()
        total_layout.addWidget(self.reforbit_frame)
        total_layout.addWidget(self.relmot_frame)
        total_layout.addWidget(self.startButton)

        self.ic_tab.setLayout(total_layout)

    def set_ic_tab_reforbit_layout(self):
        ic_layout = QHBoxLayout()

        label_row = QLabel()
        label_row.setText("Ref. inclination")
        ic_layout.addWidget(self.reference_inclination)
        ic_layout.addWidget(label_row)

        label_row = QLabel()
        label_row.setText("Ref. semi-major axis")
        ic_layout.addWidget(self.reference_semimajor_axis)
        ic_layout.addWidget(label_row)

        label_row = QLabel()
        label_row.setText("Max. position threshold")
        ic_layout.addWidget(self.maximum_distance_threshold)
        ic_layout.addWidget(label_row)

        self.reforbit_frame.setLayout(ic_layout)

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

        xy_v_heatmap = GraphWidgets.GraphViewHeatMap()
        HeatMap.heat_map_xy(xy_v_heatmap, x_v_var, y_v_var, mean_state, x_v_var, y_v_var,
                            reference_orbit, end_seconds, recorded_times)
        xy_v_heatmap_layout = QHBoxLayout()
        xy_v_heatmap_layout.addWidget(xy_v_heatmap)
        self.xy_v_heatmap_tab.setLayout(xy_v_heatmap_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    INIT_STARMAP = App()
    sys.exit(app.exec_())
