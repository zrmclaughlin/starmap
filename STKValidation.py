import numpy as np
import OrbitalElements as oe
import final_stm_targeting_tests as test
import pandas as pd
import targeting_test
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
import Transformation

r_e = 6378136.3
j2 = 1.082E-3
# mu = 3.986004415E14
mu = 3.98600436E14
k_j2 = 3*j2*mu*r_e**2 / 2


def get_rel_combined_data(state, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):
    times = np.linspace(0.0, 15000, 501)
    state = np.concatenate(([state], np.eye(len(state))), axis=0).flatten()
    t, result, target_status, stable, d_v = test.combined_targeter(times, state, times[1] - times[0], [0, 0, 0], False, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    mag = []
    for i in range(len(result[:, 0])):
        mag.append(np.linalg.norm([result[i, 7], result[i, 5], result[i, 6]]))
    # The xr axis completes the right-handed frame -> transverse (y)
    # The yr axis points to the opposite direction of the orbital angular moment - > negative normal (-z)
    # The zr axis points to the center of the Earth -> negative radial (-x)
    to_write = pd.DataFrame(data=np.asarray([-result[:, 7], result[:, 5], -result[:, 6], mag]).T)
    to_write.to_csv("chen-jing_results_test.csv")
    return t, np.asarray([-result[:, 7], result[:, 5], -result[:, 6]])


def get_rel_cw_data(state):
    times = np.linspace(0.0, 15000, 501)
    state = np.concatenate(([state], np.eye(len(state))), axis=0).flatten()
    cw_t, cw_results, target_status, stable, d_v = targeting_test.cw_propagator(times, state, times[1] - times[0], [0, 0, 0], False)
    mag = []
    for i in range(len(cw_results[:, 0])):
        mag.append(np.linalg.norm([cw_results[i, 0], cw_results[i, 1], cw_results[i, 2]]))
    to_write = pd.DataFrame(data=np.asarray([cw_results[:, 0], cw_results[:, 1], cw_results[:, 2], mag]).T)
    to_write.to_csv("cw_results_test.csv")
    return cw_t, np.asarray([cw_results[:, 0], cw_results[:, 1], cw_results[:, 2]])


def get_rel_j2_data(state):
    times = np.linspace(0.0, 15000, 501)
    state = np.concatenate(([state], np.eye(len(state))), axis=0).flatten()
    j2_t, j2_results, target_status, stable, d_v = targeting_test.j2_sedwick_propagator(times, state, times[1] - times[0], [0, 0, 0], False)
    mag = []
    for i in range(len(j2_results[:, 0])):
        mag.append(np.linalg.norm([j2_results[i, 0], j2_results[i, 1], j2_results[i, 2]]))
    to_write = pd.DataFrame(data=np.asarray([j2_results[:, 0], j2_results[:, 1], j2_results[:, 2], mag]).T)
    to_write.to_csv("j2_results_test.csv")
    return j2_t, np.asarray([j2_results[:, 0], j2_results[:, 1], j2_results[:, 2]])


def run():
    r_reference = 6629230
    a_reference = 6978137
    v_z = 0.0
    h_reference = 52673853000
    i_reference = 97.8 * np.pi / 180
    raan_reference = 250 * np.pi / 180
    theta_reference = 0.0
    w_reference = 0.0
    e_reference = 0.05

    c_d = 2.2
    a_m_reference = .01
    a_m_chaser = .007
    rho_0 = 1.438E-12
    r_0 = 6978137
    H = 109300

    x_0 = 33925
    x_0_dot = -.009431
    y_0 = 5671
    y_0_dot = -13.902
    z_0 = 85
    z_0_dot = -1.993

    wx = 0.0
    wy = -h_reference / np.linalg.norm(r_reference)**2
    wz = k_j2 * np.sin(2*i_reference) * np.sin(theta_reference) / (h_reference * np.linalg.norm(r_reference)**3)

    w_rel_frame = np.asarray([wx, wy, wz])

    p1 = x_0_dot - y_0*wz + (z_0 - np.linalg.norm(r_reference))*wy
    p2 = y_0_dot + x_0*wz
    p3 = z_0_dot + v_z - x_0*wy
    p_eq = np.asarray([p1, p2, p3])

    v_rel = np.asarray([x_0_dot, y_0_dot, z_0_dot])

    reference_orbit = oe.OrbitalElements(a_reference, e_reference, i_reference, w_reference, raan_reference, theta_reference, mu)
    print("", "reference a", reference_orbit.get_a(), "\n",
          "reference e", reference_orbit.get_e(), "\n",
          "reference i", reference_orbit.get_i(), "\n",
          "reference o", reference_orbit.get_o(), "\n",
          "reference w", reference_orbit.get_w(), "\n",
          "reference nu", reference_orbit.get_nu())

    r_reference, v_reference = reference_orbit.get_cartesian()

    print("", "Reference position", r_reference, "\n",
          "Reference position", np.linalg.norm(r_reference), "\n",
          "Reference velocity", v_reference)

    # compose dcm for transformation
    z_reference = np.asarray([-r_reference[0], -r_reference[1], -r_reference[2]]) / np.linalg.norm(np.asarray([-r_reference[0], -r_reference[1], -r_reference[2]]))
    y_reference = -1*np.cross(r_reference, v_reference) / np.linalg.norm(np.cross(r_reference, v_reference))
    x_reference = np.cross(y_reference, z_reference) / np.linalg.norm(np.cross(y_reference, z_reference))

    to_relative_motion_frame_dcm = np.asarray([x_reference, y_reference, z_reference])

    r_chaser_rel_frame = np.asarray([x_0, y_0, z_0 - np.linalg.norm(r_reference)])
    r_chaser = np.matmul(np.linalg.inv(to_relative_motion_frame_dcm), np.asarray([x_0, y_0, z_0 - np.linalg.norm(r_reference)]))
    print("R chaser inertial frame: ", r_chaser)

    p = v_rel - np.cross(w_rel_frame, r_chaser_rel_frame)
    print("P from equations", p_eq)
    print("P from frame calculations", p)
    v_chaser = np.matmul(np.linalg.inv(to_relative_motion_frame_dcm), p)
    print("Inertial chaser velocity", v_chaser)

    r_reference_rel_frame = np.matmul(to_relative_motion_frame_dcm, r_reference)
    v_reference_rel_frame = np.matmul(to_relative_motion_frame_dcm, v_reference) + np.cross(w_rel_frame, r_reference_rel_frame)

    chaser_orbit = oe.from_cartesian(r_chaser, v_chaser, mu)

    print("", "chaser a", chaser_orbit.get_a(), "\n",
          "chaser e", chaser_orbit.get_e(), "\n",
          "chaser i", chaser_orbit.get_i(), "\n",
          "chaser o", chaser_orbit.get_o(), "\n",
          "chaser w", chaser_orbit.get_w(), "\n",
          "chaser nu", chaser_orbit.get_nu())

    delta_state_0 = [np.linalg.norm(r_reference), v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3]

    # calculate j2 and cw states
    cw_frame_r = np.asarray([r_reference[0] - r_chaser[0], r_reference[1] - r_chaser[1], r_reference[2] - r_chaser[2]])
    cw_frame_v = np.asarray([v_reference[0] - v_chaser[0], v_reference[1] - v_chaser[1], v_reference[2] - v_chaser[2]])

    to_rtn_mat = np.matmul(Transformation.t_3(reference_orbit.get_nu()),
                           np.matmul(Transformation.t_1(reference_orbit.get_i()),
                           Transformation.t_3(reference_orbit.get_o())))

    w = [0.0, 0.0, np.sqrt(mu/np.linalg.norm(r_reference)**3)]
    cw_frame_r = np.matmul(to_rtn_mat, cw_frame_r)
    w_cross_r_cw = np.cross(w, cw_frame_r)
    cw_frame_v = np.matmul(to_rtn_mat, cw_frame_v)
    cw_frame_v = cw_frame_v + w_cross_r_cw

    cw_frame_state = [cw_frame_r[0], cw_frame_r[1], cw_frame_r[2], cw_frame_v[0], cw_frame_v[1], cw_frame_v[2]]

    c_t, combined_data = get_rel_combined_data(delta_state_0, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    cw_t, cw_data = get_rel_cw_data(cw_frame_state)
    j2_t, j2_data = get_rel_j2_data(cw_frame_state)

    plt.plot(c_t, combined_data[0], label="combined")
    plt.plot(c_t, cw_data[0], label="cw")
    plt.plot(c_t, j2_data[0], label="j2")
    plt.legend(loc="best")
    plt.show()


def read():
    cw_data = pd.read_csv("cw_results_test.csv")
    j2_data = pd.read_csv("j2_results_test.csv")
    combined_data = pd.read_csv("chen-jing_results_test.csv")
    stk_data = pd.read_csv("Chaser_RIC.csv")
    n = 4  # 1 for radial, 2 for in-track, 3 for cross-track
    plt.plot(cw_data.iloc[:, 0].tolist(), cw_data.iloc[:, n].tolist(), label="cw")
    plt.plot(cw_data.iloc[:, 0].tolist(), j2_data.iloc[:, n].tolist(), label="j2")
    plt.plot(cw_data.iloc[:, 0].tolist(), 1*np.asarray(combined_data.iloc[:, n].tolist()), label="combined")
    plt.plot(cw_data.iloc[:, 0].tolist(), 1000*np.asarray(stk_data.iloc[:, n].tolist()[:502]), label="stk")  # "Radial (km)"
    plt.legend(loc="best")
    plt.show()


def main():
    read()


if __name__ == "__main__":
    main()
