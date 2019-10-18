import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from itertools import cycle
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
plt.style.use("seaborn")
matplotlib.rcParams.update({'font.size': 8})
from mpl_toolkits import mplot3d

r_e = 6378136.3
j2 = 1082.626E-6
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2

############### EOMs ####################

def chen_jing_eom(t, state, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):
    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3
    wy = -state[2] / state[0]**2
    wz = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)

    r_chaser = np.linalg.norm([state[5], state[6], state[7] - state[0]])
    z_chaser = state[5]*np.cos(state[3])*np.sin(state[4]) - state[6]*np.cos(state[4]) - (state[7] - state[0])*np.sin(state[4])*np.sin(state[3])
    w_bar = -mu / r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*z_chaser**2 / r_chaser**7
    zeta = 2*k_j2*z_chaser / r_chaser**5

    v_reference = np.linalg.norm([state[2]/state[0], 0, state[1]])
    rho_reference = rho_0*np.exp(-(state[0] - r_0)/H)
    f_drag_reference = - .5*c_d*a_m_reference*rho_reference

    state_size = len(state)  # 1: implies x ODEs
    dstate_dt = np.zeros((1, state_size))

    dstate_dt[0][0] = -state[1]  # d r / dt
    dstate_dt[0][1] = mu / state[0]**2 - state[2]**2 / state[0]**3 + k_j2*(1 - 3*np.sin(state[4])**2 * np.sin(state[3])**2) / state[0]**4 + f_drag_reference*state[1]*v_reference  # d v_z / dt
    dstate_dt[0][2] = -k_j2*np.sin(state[4])**2*np.sin(2*state[3]) / state[0]**3  # d h_reference / dt
    dstate_dt[0][3] = state[2] / state[0]**2 + 2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2] * state[0]**3)  # d theta_reference / dt
    dstate_dt[0][4] = -k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**3)  # d i_reference / dt
    dstate_dt[0][5] = state[8] + state[6]*wz - (state[7] - state[0])*wy  # d x / dt
    dstate_dt[0][6] = state[9] - state[5]*wz  # d y / dt
    dstate_dt[0][7] = state[10] - state[1] + state[5]*wy  # d z / dt

    v_chaser = [dstate_dt[0][5] - state[6]*wz + (state[7] - state[0])*wy, dstate_dt[0][6] + state[5]*wz, dstate_dt[0][7] + state[1] - state[5]*wy]
    rho_chaser = rho_0*np.exp(-(r_chaser - r_0)/H)
    f_drag_chaser = - .5*c_d*a_m_chaser*rho_chaser*np.linalg.norm(v_chaser)

    dstate_dt[0][8] = w_bar*state[5] - zeta*np.cos(state[3])*np.sin(state[4]) + state[9]*wz - state[10]*wy + f_drag_chaser*v_chaser[0]  # d p1 / dt
    dstate_dt[0][9] = w_bar*state[6] + zeta*np.cos(state[4]) - state[8]*wz + f_drag_chaser*v_chaser[1]  # d p2 / dt
    dstate_dt[0][10] = w_bar*(state[7] - state[0]) + zeta*np.sin(state[3])*np.sin(state[4]) + state[8]*wy + f_drag_chaser*v_chaser[2]  # d p3 / dt
    return dstate_dt


def j2_drag_ecc_propagator(state_0, time, number_of_points, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    time_vector = np.linspace(0.0, time, number_of_points)
    step = time_vector[1] - time_vector[0]
    sc = sp.integrate.ode(lambda t, x: chen_jing_eom(t, x, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)).set_integrator('dopri5',
                                                                            atol=1e-10,
                                                                            rtol=1e-5)
    sc.set_initial_value(state_0, )
    t = np.zeros((len(time_vector), len(state_0)))
    result = np.zeros((len(time_vector), len(state_0)))
    t[0] = time_vector[0]
    result[0][:] = state_0
    step_count = 1

    while sc.successful() and step_count < len(t):
        sc.integrate(sc.t + step)
        t[step_count] = sc.t
        result[step_count][:] = sc.y
        step_count += 1

    return time_vector, [result[:, 5], result[:, 6], result[:, 7]]


def main():
    r_reference = 6629230
    v_z = 0.0
    h_reference = 52673853000
    i_reference = 97.8 * np.pi / 180
    raan_reference = 250 * np.pi / 180
    theta_reference = 0.0 * np.pi / 180
    e_reference = 0.05

    c_d = 2.2
    a_m_reference = .01
    a_m_chaser = .007
    rho_0 = 1.438E-12
    r_0 = 6978137
    H = 109300

    x_0 = 33925
    x_0_dot = -.009431
    y_0 =  5671
    y_0_dot = -13.902
    z_0 = 85
    z_0_dot = -1.993

    wy = -h_reference / r_reference**2
    wz = k_j2 * np.sin(2*i_reference) * np.sin(theta_reference) / (h_reference * r_reference**3)

    p1 = x_0_dot - y_0*wz + (z_0 - r_reference)*wy
    p2 = y_0_dot + x_0*wz
    p3 = z_0_dot + v_z - x_0*wy

    state = [r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3]

    number_of_steps = 40000
    time = 4000
    recorded_times, results = j2_drag_ecc_propagator(state, time, number_of_steps, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    ax = plt.axes(projection='3d')
    ax.set_xlabel("Radial")
    ax.set_ylabel("In-Track")
    ax.set_zlabel("Cross-Track")
    ax.set_title("Relative Motion for " + str(time) + " seconds")

    # Data for a three-dimensional line
    ax.plot3D(results[0], results[1], results[2])
    # plt.plot(time, results[2])
    plt.show()

    return


if __name__ == "__main__":
    main()
