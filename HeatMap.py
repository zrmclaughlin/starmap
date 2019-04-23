import OrbitalElements
import J2RelativeMotion
import GraphWidgets
import numpy as np


# ############################## HEAT MAP GENERATOR ############################## #

def heat_map_xy(GraphViewHeatMap, x_variance, y_variance, mean_state, vary_size_x, vary_size_y, reference_orbit, end_seconds, recorded_times):
    vary_size_x = 30
    vary_size_y = 30
    x_variance = np.linspace(-x_variance, x_variance, vary_size_x).tolist()
    y_variance = np.linspace(-y_variance, y_variance, vary_size_y).tolist()

    success_level = np.zeros([vary_size_x, vary_size_y])
    times = np.linspace(0.0, end_seconds, recorded_times)

    step = times[1] - times[0]

    for i in range(vary_size_x):

        for j in range(vary_size_y):

            # x0 = [x_variance[i], y_variance[j], mean_state[2], mean_state[3], mean_state[4], mean_state[5]]
            x0 = [mean_state[0], mean_state[1], mean_state[2], x_variance[i], y_variance[j], mean_state[5]]

            success_level[i][j] = J2RelativeMotion.j2_sedwick_propagator(x0, reference_orbit, times, step, False)\
                                  / recorded_times * 100

    map_x, map_y = np.array(np.meshgrid(x_variance, y_variance))

    # print(np.amax(success_level))

    max_x, max_y = np.where(np.asarray(success_level) == np.asarray(success_level).max())
    # print(max_x, max_y)
    max_percentage = success_level[max_x[0]][max_x[1]]
    # print(max_percentage)

    axis_labels = ["Radial Variance (m/s)", "In-Track Variance (m/s)", "Percentage of Time within Constraint"]
    title = "Relative Motion Heatmap for " + str(end_seconds) + " seconds"

    GraphViewHeatMap.update_graph(map_x, map_y, success_level, title, axis_labels)


# ################################################################################ #