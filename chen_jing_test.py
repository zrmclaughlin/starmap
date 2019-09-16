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
j2 = 1.082626E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2

############## helpers ###################

def get_rho(rho_0, r, r_0, H):
    return rho_0*np.exp(-(r - r_0)/H)

def get_r_chaser(x, y, z, r_reference):
    return [x, y, z - r_reference]

def get_raan_dot(i_reference, theta, h, r):
    return -2*k_j2*np.cos(i_reference)*np.sin(theta)**2 / h / r**3

def get_i_dot(i_reference, theta, h, r):
    return -k_j2*np.sin(2*i_reference)*np.sin(2*theta) / 2 / h / r**3

def get_theta_dot(i_reference, theta, h, r):
    return h**2 / r + 2*k_j2*np.cos(i_reference)**2 * np.sin(theta)**2 / h / r**3

def get_Z_chaser(x, y, z, r_reference, theta, i_reference):
    return x*np.cos(theta)*np.sin(i_reference) - y*np.cos(i_reference) - (z - r_reference)*np.sin(theta)*np.sin(i_reference)

def get_zeta(Z_chaser, r_chaser):
    return 2*k_j2*Z_chaser / r_chaser**5

def get_w_bar(Z_chaser, r_chaser):
    return -mu/r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*Z_chaser**2 / r_chaser**7

def get_w_reference(h_reference, r_reference, i_reference, theta_reference):
    return [0, -h_reference/r_reference**2, k_j2*np.sin(2*i_reference)*np.sin(theta_reference) / h_reference / r_reference**2]

############### EOMs ####################

def chen_jing_eom(t, state, rho_0, H, r_0, c_d, a_m_chaser, a_m_reference):
    # <- v_z, r_reference, h_reference, raan_reference, i_reference, theta_reference,
    #    x, y, z, p1, p2, p3
    v_reference = [state[2]/state[1], 0, state[0]]
    r_chaser = get_r_chaser(state[6], state[7], state[8], state[1])
    w_reference = get_w_reference(state[2], state[1], state[4], state[5])
    Z_chaser = get_Z_chaser(state[6], state[7], state[8], state[1], state[5], state[4])
    w_bar = get_w_bar(Z_chaser, np.linalg.norm(r_chaser))
    zeta = get_zeta(Z_chaser, np.linalg.norm(r_chaser))

    state_size = len(state)  # 1: implies x ODEs
    dstate_dt = np.zeros((1, state_size))

    rho_reference = get_rho(rho_0, state[1], r_0, H)

    dstate_dt[0][0] = mu / state[1]**2 - state[2]**2 / state[1]**3 + k_j2 / state[1]**4 * (1 - 3*np.sin(state[4])**2 * np.sin(state[5])**2) - .5*rho_reference*c_d*a_m_reference*state[0]*np.linalg.norm(v_reference)  # d v_z / dt
    dstate_dt[0][1] = - state[0] # d r / dt
    dstate_dt[0][2] = - k_j2 * np.sin(state[4])**2 * np.sin(2*state[5])**2 - .5 * rho_reference * c_d * a_m_reference * state[0] * np.linalg.norm(v_reference)  # d h_reference / dt
    dstate_dt[0][3] = - 2 * k_j2 * np.cos(state[4])*np.sin(state[5])  # d raan_reference / dt
    dstate_dt[0][4] = - k_j2 * np.sin(2*state[5])*np.sin(2*state[4]) / 2 / state[2] / state[1]**3  # d i_reference / dt
    dstate_dt[0][5] = state[2] / state[1]**2 + 2 * k_j2 * np.cos(state[4])**2*np.sin(state[5])**2 / state[2] / state[1] ** 2  # d theta_reference / dt
    dstate_dt[0][6] = state[9] + state[7]*w_reference[2] - (state[8] - state[1])*w_reference[1]  # d x / dt
    dstate_dt[0][7] = state[10] - state[6]*w_reference[2]  # d y / dt
    dstate_dt[0][8] = state[11] - state[0] + state[6]*w_reference[1]  # d z / dt
    v_chaser = [dstate_dt[0][6] - state[7]*w_reference[2] + (state[8] - state[1])*w_reference[1],
                dstate_dt[0][7] + state[6]*w_reference[2],
                dstate_dt[0][8] + state[0] - state[6]*w_reference[1]]
    rho_chaser = get_rho(rho_0, np.linalg.norm(r_chaser), r_0, H)
    f_drag_constant = -.5*rho_chaser*c_d*a_m_chaser*np.linalg.norm(v_chaser)
    dstate_dt[0][9] = w_bar * state[6] - zeta*np.cos(state[5])*np.sin(state[1]) + state[10]*w_reference[2] - state[11]*w_reference[1] + f_drag_constant*(dstate_dt[0][6] - state[7]*w_reference[2] + (state[8] - state[1])*w_reference[1])  # d p1 / dt
    dstate_dt[0][10] = w_bar * state[7] + zeta*np.cos(state[4]) + state[9]*w_reference[2] + f_drag_constant*(dstate_dt[0][7] + w_reference[2]*state[6])  # d p2 / dt
    dstate_dt[0][11] = w_bar * (state[8] - state[1]) + zeta*np.sin(state[5])*np.sin(state[4]) + state[9]*w_reference[1] + f_drag_constant*(dstate_dt[0][8] + state[0] - state[6]*w_reference[1])  # d p3 / dt
    return dstate_dt


def j2_drag_ecc_propagator(state_0, rho_0, H, r_0, c_d, a_m_chaser, a_m_reference, time, step):
    sc = sp.integrate.ode(lambda t, x: chen_jing_eom(t, x, rho_0, H, r_0, c_d, a_m_chaser, a_m_reference)).set_integrator('dopri5', atol=1e-10,
                                                                                          rtol=1e-10)
    sc.set_initial_value(state_0, time[0])
    t = np.zeros((len(time), len(state_0)))
    result = np.zeros((len(time), len(state_0)))
    t[0] = time[0]
    result[0][:] = state_0
    step_count = 1

    while sc.successful() and step_count < len(t):
        sc.integrate(sc.t + step)
        t[step_count] = sc.t
        result[step_count][:] = sc.y
        step_count += 1

    return [result[:, 6], result[:, 7], result[:, 8]]


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
    p1 = 7946
    y_0 =  5671
    y_0_dot = -13.902
    p2 = -13.9
    z_0 = 85
    z_0_dot = -1.993
    p3 = 38.7
    state = [v_z, r_reference, h_reference, raan_reference, i_reference, theta_reference,
             x_0, y_0, z_0, p1, p2, p3]
    end_seconds = 10000
    steps = 500
    time = np.linspace(0, end_seconds, steps)
    # print(time)
    results = j2_drag_ecc_propagator(state, rho_0, H, r_0, c_d, a_m_chaser, a_m_reference, time, steps)
    ax = plt.axes(projection='3d')
    ax.set_xlabel("Radial")
    ax.set_ylabel("In-Track")
    ax.set_zlabel("Cross-Track")
    ax.set_title("Relative Motion for " + str(end_seconds) + " seconds")

    # Data for a three-dimensional line
    ax.plot3D(results[0], results[1], results[2])
    plt.show()

    return


if __name__ == "__main__":
    main()
