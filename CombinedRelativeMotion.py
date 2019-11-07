import numpy as np
import scipy as sp
from scipy import integrate
import CombinedModelJacobian as CJJacobian
import TargetingUtils
from mpmath import *

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2


def chen_jing_eom_st(t, state, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3
    state_size = len(state)  # 1: implies x ODEs
    S_T = TargetingUtils.recompose(state, len(A[0]))
    S_T_dt = np.matmul(A(state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10]), S_T)
    wy = -state[2] / state[0]**2
    wz = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)

    r_chaser = np.linalg.norm([state[5], state[6], state[7] - state[0]])
    z_chaser = state[5]*np.cos(state[3])*np.sin(state[4]) - state[6]*np.cos(state[4]) - (state[7] - state[0])*np.sin(state[4])*np.sin(state[3])
    w_bar = -mu / r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*z_chaser**2 / r_chaser**7
    zeta = 2*k_j2*z_chaser / r_chaser**5

    v_reference = np.linalg.norm([state[2]/state[0], 0, state[1]])
    rho_reference = rho_0*np.exp(-(state[0] - r_0)/H)
    f_drag_reference = - .5*c_d*a_m_reference*rho_reference

    dstate_dt = np.asarray([])

    dstate_dt[0] = -state[1]  # d r / dt
    dstate_dt[1] = mu / state[0]**2 - state[2]**2 / state[0]**3 + k_j2*(1 - 3*np.sin(state[4])**2 * np.sin(state[3])**2) / state[0]**4 + f_drag_reference*state[1]*v_reference  # d v_z / dt
    dstate_dt[2] = -k_j2*np.sin(state[4])**2*np.sin(2*state[3]) / state[0]**3  # d h_reference / dt
    dstate_dt[3] = state[2] / state[0]**2 + 2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2] * state[0]**3)  # d theta_reference / dt
    dstate_dt[4] = -k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**3)  # d i_reference / dt
    dstate_dt[5] = state[8] + state[6]*wz - (state[7] - state[0])*wy  # d x / dt
    dstate_dt[6] = state[9] - state[5]*wz  # d y / dt
    dstate_dt[7] = state[10] - state[1] + state[5]*wy  # d z / dt

    v_chaser = [dstate_dt[5] - state[6]*wz + (state[7] - state[0])*wy, dstate_dt[6] + state[5]*wz, dstate_dt[7] + state[1] - state[5]*wy]
    rho_chaser = rho_0*np.exp(-(r_chaser - r_0)/H)
    f_drag_chaser = - .5*c_d*a_m_chaser*rho_chaser*np.linalg.norm(v_chaser)

    dstate_dt[8] = w_bar*state[5] - zeta*np.cos(state[3])*np.sin(state[4]) + state[9]*wz - state[10]*wy + f_drag_chaser*v_chaser[0]  # d p1 / dt
    dstate_dt[9] = w_bar*state[6] + zeta*np.cos(state[4]) - state[8]*wz + f_drag_chaser*v_chaser[1]  # d p2 / dt
    dstate_dt[10] = w_bar*(state[7] - state[0]) + zeta*np.sin(state[3])*np.sin(state[4]) + state[8]*wy + f_drag_chaser*v_chaser[2]  # d p3 / dt

    dstate_dt = np.concatenate(([dstate_dt], S_T_dt), axis=0).flatten()

    return dstate_dt

def chen_jing_targeter(state_0, targeted_state, time, step, end_seconds, thresh_min,
                        thresh_max, target_status, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # compute constants for reuse in targeting at time t=0
    # some variables are assigned from the initial state vector for clarity's sake
    wy_0 = -state_0[2] / state_0[0] ** 2
    wz_0 = k_j2 * np.sin(2 * state_0[4]) * np.sin(state_0[3]) / (state_0[2] * state_0[0] ** 3)
    r_reference_0 = state_0[0], v_z_0 = state_0[1]
    x_0 = state_0[5], y_0 = state_0[6], z_0 = state_0[7]
    p1_0 = state_0[8], p2_0 = state_0[9], p3_0 = state_0[10]
    v_0 = velocity_from_state(wy_0, wz_0, r_reference_0, v_z_0, x_0, y_0, z_0, p1_0, p2_0, p3_0)

    # Jacobian matrix
    A = CJJacobian.get_jacobian(c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3

    sc = sp.integrate.ode(lambda t, x: chen_jing_eom_st(t, state_0, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)).\
        set_integrator('dopri5', atol=1e-10, rtol=1e-5)
    sc.set_initial_value(state_0, time[0])

    results = [[], [], [], [], [], []]
    t = []

    dp = [0.0, 0.0, 0.0]

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

            # compute inverse of the state transition matrix
            S_T_inv = np.linalg.inv(TargetingUtils.recompose(sc.y, 11))
            # substitute ideal positions at time = k
            modified_state_time_k = np.asarray([sc.y[0], sc.y[1], sc.y[2], sc.y[3], sc.y[4],
                           targeted_state[0], targeted_state[1], targeted_state[2],
                           sc.y[8], sc.y[9], sc.y[10]])
            # compute altered state at time = 0
            modified_state_time_0 = np.matmul(S_T_inv, modified_state_time_k)
            # select out the values for the canonical variables we're interested in changing
            dp = [modified_state_time_0[8], modified_state_time_0[9], modified_state_time_0[10]]

            target_status = False
            stable = False

    return dp, results, current_time, target_status

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


def velocity_from_state(wy, wz, r_reference, v_z, x, y, z, p1, p2, p3):
    vx = p1 + y*wz - (z - r_reference)*wy
    vy = p2 - x*wz
    vz = p3 - v_z + x*wy
    return [vx, vy, vz]
