import numpy as np
import scipy as sp
from scipy import integrate
import TargetingUtils
from mpmath import *

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2


def chen_jing_eom_st(t, state, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):
    return

def chen_jing_targeter(state, nominal_formation, reference_orbit, time, step, end_seconds, thresh_min,
                        thresh_max, target_status, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    wy = -state[2] / state[0]**2
    wz = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)

    r_chaser = np.linalg.norm([state[5], state[6], state[7] - state[0]])
    z_chaser = state[5]*np.cos(state[3])*np.sin(state[4]) - state[6]*np.cos(state[4]) - (state[7] - state[0])*np.sin(state[4])*np.sin(state[3])
    w_bar = -mu / r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*z_chaser**2 / r_chaser**7
    zeta = 2*k_j2*z_chaser / r_chaser**5

    v_reference = np.linalg.norm([state[2]/state[0], 0, state[1]])
    rho_reference = rho_0*np.exp(-(state[0] - r_0)/H)
    f_drag_reference = - .5*c_d*a_m_reference*rho_reference

    # Jacobian matrix
    A = np.zeros(shape=(11, 11))
    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3
    A[0][0] = 0
    A[0][1] = -1
    A[0][2] = 0
    A[0][3] = 0
    A[0][4] = 0
    A[0][5] = 0
    A[0][6] = 0
    A[0][7] = 0
    A[0][8] = 0
    A[0][9] = 0
    A[0][10] = 0

    # d v_z / dt = mu / state[0]**2 - state[2]**2 / state[0]**3 + k_j2*(1 - 3*np.sin(state[4])**2 * np.sin(state[3])**2) / state[0]**4 + f_drag_reference*state[1]*v_reference
    A[1][0] = -2 / state[0]**3 + 3*state[2]**2 / state[0]**4 - 4*k_j2 / state[0]**5*(1 - 3*np.sin(state[4])**2 * np.sin(state[3])**2)  # dr
    A[1][1] = 2*f_drag_reference*state[1]  # d v_z
    A[1][2] = - 2*state[2] / state[0]**3  # d h
    A[1][3] = -k_j2*2*3*np.sin(state[4])**2*np.cos(state[3])*np.sin(state[3]) / state[0]**4  # d theta
    A[1][4] = -k_j2*2*3*np.sin(state[4])*np.cos(state[4])*np.sin(state[3])**2 / state[0]**4  # d i
    A[1][5] = 0  # d x
    A[1][6] = 0  # d y
    A[1][7] = 0  # d z
    A[1][8] = 0  # d p1
    A[1][9] = 0  # d p2
    A[1][10] = 0  # d p3

    # d h / dt = -k_j2*np.sin(state[4])**2*np.sin(2*state[3]) / state[0]**3
    A[2][0] = 3*k_j2*np.sin(state[4])**2*np.sin(2*state[3]) / state[0]**4  # dr
    A[2][1] = 0  # d v_z
    A[2][2] = 0  # d h
    A[2][3] = -k_j2*np.sin(state[4])**2*2*np.cos(2*state[3]) / state[0]**3  # d theta
    A[2][4] = -k_j2*2*np.sin(state[4])*np.cos(state[4])*np.sin(2*state[3]) / state[0]**3  # d i
    A[2][5] = 0  # d x
    A[2][6] = 0  # d y
    A[2][7] = 0  # d z
    A[2][8] = 0  # d p1
    A[2][9] = 0  # d p2
    A[2][10] = 0  # d p3

    # d theta / dt = state[2] / state[0]**2 + 2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2] * state[0]**3)  # d theta_reference / dt
    A[3][0] = -2*state[2] / state[0]**3 - 3*2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2] * state[0]**4) # dr
    A[3][1] = 0  # d v_z
    A[3][2] = 1 / state[0]**2 - 2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2]**2 * state[0]**3)   # d h
    A[3][3] = 2*k_j2*np.cos(state[4])**2*2*np.sin(state[3])*np.cos(state[3]) / (state[2] * state[0]**3)  # d theta
    A[3][4] = -2*2*k_j2*np.cos(state[4])*np.sin(state[4])*np.sin(state[3])**2 / (state[2] * state[0]**3)  # d i
    A[3][5] = 0  # d x
    A[3][6] = 0  # d y
    A[3][7] = 0  # d z
    A[3][8] = 0  # d p1
    A[3][9] = 0  # d p2
    A[3][10] = 0  # d p3

    # d i_reference / dt = -k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**3)
    A[4][0] = 3*k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**4)  # dr
    A[4][1] = 0  # d v_z
    A[4][2] = k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2]**2 * state[0]**3)   # d h
    A[4][3] = -k_j2*2*np.cos(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**3)  # d theta
    A[4][4] = -k_j2*np.sin(2*state[3])*2*np.cos(2*state[4]) / (2 * state[2] * state[0]**3)  # d i
    A[4][5] = 0  # d x
    A[4][6] = 0  # d y
    A[4][7] = 0  # d z
    A[4][8] = 0  # d p1
    A[4][9] = 0  # d p2
    A[4][10] = 0  # d p3

    # d x / dt = state[8] + state[6]*wz - state[7]*wy  + state[0]*wy
    A[5][0] = -3*state[6]*k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**4) - state[7]*(2*state[2] / state[0]**3) + state[2] / state[0]**2# dr
    A[5][1] = 0  # d v_z
    A[5][2] = -state[6]*k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2]**2 * state[0]**3) - (state[7] - state[0]) / state[0]**2    # d h
    A[5][3] = state[6]*k_j2 * np.sin(2*state[4]) * np.cos(state[3]) / (state[2] * state[0]**3)  # d theta
    A[5][4] = state[6]*k_j2 * 2*np.cos(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)  # d i
    A[5][5] = 0  # d x
    A[5][6] = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)  # d y
    A[5][7] = state[2] / state[0]**2 # d z
    A[5][8] = 1  # d p1
    A[5][9] = 0  # d p2
    A[5][10] = 0  # d p3

    # d y / dt = state[9] - state[5]*wz
    A[6][0] = 3*state[5]*k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**4)  # dr
    A[6][1] = 0  # d v_z
    A[6][2] = state[5]*k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2]**2 * state[0]**3)  # d h
    A[6][3] = - state[5]*k_j2 * np.sin(2*state[4]) * np.cos(state[3]) / (state[2] * state[0]**3)  # d theta
    A[6][4] = - state[5]*k_j2 * 2*np.cos(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)  # d i
    A[6][5] = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)  # d x
    A[6][6] = 0  # d y
    A[6][7] = 0  # d z
    A[6][8] = 0  # d p1
    A[6][9] = 1  # d p2
    A[6][10] = 0  # d p3

    # d z / dt = state[10] - state[1] + state[5]*wy
    A[7][0] = state[5]*(2*state[2] / state[0]**3)  # dr
    A[7][1] = -1  # d v_z
    A[7][2] = state[5]*(-1 / state[0]**2)  # d h
    A[7][3] = 0  # d theta
    A[7][4] = 0  # d i
    A[7][5] = -state[2] / state[0]**2  # d x
    A[7][6] = 0  # d y
    A[7][7] = 0  # d z
    A[7][8] = 0  # d p1
    A[7][9] = 0  # d p2
    A[7][10] = 1  # d p3

    xdot = state[8] + state[6] * wz - (state[7] - state[0]) * wy  # d x / dt
    ydot = state[9] - state[5] * wz  # d y / dt
    zdot = state[10] - state[1] + state[5] * wy  # d z / dt

    v_chaser = [xdot - state[6] * wz + (state[7] - state[0]) * wy,
                ydot + state[5] * wz,
                zdot + state[1] - state[5] * wy]
    rho_chaser = rho_0 * np.exp(-(r_chaser - r_0) / H)
    f_drag_chaser = - .5 * c_d * a_m_chaser * rho_chaser * np.linalg.norm(v_chaser)

    # d p1 / dt = w_bar*state[5] - zeta*np.cos(state[3])*np.sin(state[4]) + state[9]*wz - state[10]*wy + f_drag_chaser*v_chaser[0]

    A[8][0] = -3*state[9]*(np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**4)) + state[10]*(2*state[2] / state[0]**3) + \
              f_drag_chaser*(state[6]*(np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**4)) - state[7]*(2*state[2] / state[0]**3) + state[2] / state[0]**2)  # dr
    A[8][1] = 0   # d v_z
    A[8][2] = 0  # d h
    A[8][3] = 0  # d theta
    A[8][4] = 0  # d i
    A[8][5] = 0  # d x
    A[8][6] = 0  # d y
    A[8][7] = 0  # d z
    A[8][8] = 0  # d p1
    A[8][9] = 0  # d p2
    A[8][10] = 0  # d p3

    # d p2 / dt = w_bar*state[6] + zeta*np.cos(state[4]) - state[8]*wz + f_drag_chaser*v_chaser[1]
    A[9][0] = 0  # dr
    A[9][1] = 0  # d v_z
    A[9][2] = 0  # d h
    A[9][3] = 0  # d theta
    A[9][4] = 0  # d i
    A[9][5] = 0  # d x
    A[9][6] = 0  # d y
    A[9][7] = 0  # d z
    A[9][8] = 0  # d p1
    A[9][9] = 0  # d p2
    A[9][10] = 0  # d p3

    # d p3 / dt = w_bar*(state[7] - state[0]) + zeta*np.sin(state[3])*np.sin(state[4]) + state[8]*wy + f_drag_chaser*v_chaser[2]
    A[10][0] = 0  # dr
    A[10][1] = 0  # d v_z
    A[10][2] = 0  # d h
    A[10][3] = 0  # d theta
    A[10][4] = 0  # d i
    A[10][5] = 0  # d x
    A[10][6] = 0  # d y
    A[10][7] = 0  # d z
    A[10][8] = 0  # d p1
    A[10][9] = 0  # d p2
    A[10][10] = 0  # d p3

    sc = sp.integrate.ode(lambda t, x: chen_jing_eom_st(t, state, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)).set_integrator('dopri5', atol=1e-10,
                                                                                          rtol=1e-5)
    sc.set_initial_value(state, time[0])

    results = [[], [], [], [], [], []]
    t = []

    dv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    stable = True
    current_time = 0
    target_status = True

    while sc.successful() and stable and current_time < end_seconds:
        sc.integrate(sc.t + step)
        current_time = current_time + step
        t.append(current_time)
        results[0].append(sc.y[5])
        results[1].append(sc.y[6])
        results[2].append(sc.y[7])
        results[3].append(sc.y[8])
        results[4].append(sc.y[9])
        results[5].append(sc.y[10])
        if np.sqrt((sc.y[5]**2 + sc.y[6]**2 + sc.y[7]**2)) > thresh_max:  # do targeting!
            # S_T_vv_inv = np.linalg.inv(TargetingUtils.get_S_T_vv(sc.y))
            # S_T_rv_inv = np.linalg.inv(TargetingUtils.get_S_T_rv(sc.y))
            # # determine a maneuver to put the spacecraft back on track :)
            # dv1 = np.matmul(S_T_rv_inv, np.asarray([nominal_formation[0], nominal_formation[1], nominal_formation[2]]))
            # dv2 = np.matmul(S_T_vv_inv, np.asarray([nominal_formation[3], nominal_formation[4], nominal_formation[5]]))
            # dv = [0, 0, 0, dv1[0, 0], dv1[0, 1], dv1[0, 2]]
            target_status = False
            stable = False

    return dv, results, current_time, target_status

# ############################################################################ #


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


def j2_drag_ecc_propagator(state_0, time, number_of_points, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H, type, thresh_min, thresh_max, target_status):

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

    if type == 0:
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

        return [result[:, 5], result[:, 6], result[:, 7]]

    elif type == 1:  # if we are counting how many fit within a threshold rather than simply propagating

        success_count = 0
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1
            if (np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) > thresh_min) & (
                        np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) < thresh_max):
                success_count += 1

        return success_count

    elif type == 2:
        result_in_range = [[], [], []]
        t_in_range = []
        result_out_of_range = [[], [], []]
        t_out_of_range = []
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            step_count += 1
            if (np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) > thresh_max) | \
                    (np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) < thresh_min):
                result_out_of_range[0].append(sc.y[5])
                result_out_of_range[1].append(sc.y[6])
                result_out_of_range[2].append(sc.y[7])
                t_out_of_range.append(sc.t)
            else:
                result_in_range[0].append(sc.y[5])
                result_in_range[1].append(sc.y[6])
                result_in_range[2].append(sc.y[7])
                t_in_range.append(sc.t)

        return t_in_range, result_in_range, t_out_of_range, result_out_of_range

    elif type == 3:  # to get amounts of time for each pass
        pass_lengths = []
        pass_on = False
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1
            if (np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) > thresh_min) & (
                        np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2) < thresh_max):
                if not pass_on:
                    pass_on = True
                    pass_lengths.append(0)
                else:
                    pass_lengths[-1] += 1
            else:
                pass_on = False

        return pass_lengths

    elif type == 4:
        magnitudes = []
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            step_count += 1
            magnitudes.append(np.sqrt(sc.y[5] ** 2 + sc.y[6] ** 2 + sc.y[7] ** 2))

        return magnitudes

    # return time_vector, [result[:, 5], result[:, 6], result[:, 7]]