import numpy as np
import scipy as sp
from scipy import integrate
import TargetingUtils
from mpmath import *

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14

# ######################## TARGETING SEDWICK J2 REOM ######################### #


def sedwick_eom_st(t, delta_state, A, n, c, l, q, phi):
    state_size = len(delta_state)  # 1: implies x ODEs
    dx_dt = np.zeros((1, state_size))
    S_T = TargetingUtils.recompose(delta_state, len(A[0]))
    S_T_dt = np.matmul(A, S_T).tolist()

    dx_dt[0][0] = delta_state[3]
    dx_dt[0][1] = delta_state[4]
    dx_dt[0][2] = delta_state[5]
    dx_dt[0][3] = 2 * n * c * delta_state[4] + (5 * c ** 2 - 2) * n ** 2 * delta_state[0]
    dx_dt[0][4] = -2 * n * c * delta_state[3]
    dx_dt[0][5] = -q ** 2 * delta_state[2] + 2 * l * q * np.cos(q * t + phi)

    dx_dt[0][6] = S_T_dt[0][0]
    dx_dt[0][7] = S_T_dt[0][1]
    dx_dt[0][8] = S_T_dt[0][2]
    dx_dt[0][9] = S_T_dt[0][3]
    dx_dt[0][10] = S_T_dt[0][4]
    dx_dt[0][11] = S_T_dt[0][5]
    dx_dt[0][12] = S_T_dt[1][0]
    dx_dt[0][13] = S_T_dt[1][1]
    dx_dt[0][14] = S_T_dt[1][2]
    dx_dt[0][15] = S_T_dt[1][3]
    dx_dt[0][16] = S_T_dt[1][4]
    dx_dt[0][17] = S_T_dt[1][5]
    dx_dt[0][18] = S_T_dt[2][0]
    dx_dt[0][19] = S_T_dt[2][1]
    dx_dt[0][20] = S_T_dt[2][2]
    dx_dt[0][21] = S_T_dt[2][3]
    dx_dt[0][22] = S_T_dt[2][4]
    dx_dt[0][23] = S_T_dt[2][5]
    dx_dt[0][24] = S_T_dt[3][0]
    dx_dt[0][25] = S_T_dt[3][1]
    dx_dt[0][26] = S_T_dt[3][2]
    dx_dt[0][27] = S_T_dt[3][3]
    dx_dt[0][28] = S_T_dt[3][4]
    dx_dt[0][29] = S_T_dt[3][5]
    dx_dt[0][30] = S_T_dt[4][0]
    dx_dt[0][31] = S_T_dt[4][1]
    dx_dt[0][32] = S_T_dt[4][2]
    dx_dt[0][33] = S_T_dt[4][3]
    dx_dt[0][34] = S_T_dt[4][4]
    dx_dt[0][35] = S_T_dt[4][5]
    dx_dt[0][36] = S_T_dt[5][0]
    dx_dt[0][37] = S_T_dt[5][1]
    dx_dt[0][38] = S_T_dt[5][2]
    dx_dt[0][39] = S_T_dt[5][3]
    dx_dt[0][40] = S_T_dt[5][4]
    dx_dt[0][41] = S_T_dt[5][5]
    return dx_dt

# ############################## SEDWICK J2 REOM ############################## #

def sedwick_eom(t, delta_state, n, c, l, q, phi):
    state_size = len(delta_state)  # 1: implies x ODEs
    dx_dt = np.zeros((1, state_size))
    dx_dt[0][0] = delta_state[3]
    dx_dt[0][1] = delta_state[4]
    dx_dt[0][2] = delta_state[5]
    dx_dt[0][3] = 2 * n * c * delta_state[4] + (5 * c ** 2 - 2) * n ** 2 * delta_state[0]
    dx_dt[0][4] = -2 * n * c * delta_state[3]
    dx_dt[0][5] = -q ** 2 * delta_state[2] + 2 * l * q * np.cos(q * t + phi)
    return dx_dt


def evaluate_j2_constants(reference_orbit, delta_state_0):
    # calculate j2 parameter effects, assuming that the reference
    # orbit is the same as satellite 1's circularized orbit.
    s = 3 * j2 * r_e ** 2 / (8 * reference_orbit.get_a() ** 2) * (1 + 3 * np.cos(2 * reference_orbit.get_i()))
    c = np.sqrt(s + 1)
    n = np.sqrt(mu / reference_orbit.get_a() ** 3)
    k = n * c + 3 * n * j2 * r_e ** 2 / (2 * reference_orbit.get_a() ** 2) * np.cos(reference_orbit.get_i()) ** 2

    i_sat2 = reference_orbit.get_i() - delta_state_0[5] / (k * reference_orbit.get_a())

    delta_RAAN_0 = delta_state_0[2] / (reference_orbit.get_a() * np.sin(reference_orbit.get_i()))
    gamma_0 = float(acot((cot(i_sat2) * np.sin(reference_orbit.get_i()) - np.cos(reference_orbit.get_i()) * np.cos(
        delta_RAAN_0)) / np.sin(delta_RAAN_0)))
    phi_0 = np.arccos(
        np.cos(reference_orbit.get_i()) * np.cos(i_sat2) + np.sin(reference_orbit.get_i()) * np.sin(i_sat2) * np.cos(
            delta_RAAN_0))

    d_RAAN_sat1_0 = -3 * n * j2 * r_e ** 2 / (2 * reference_orbit.get_a() ** 2) * np.cos(reference_orbit.get_i())
    d_RAAN_sat2_0 = -3 * n * j2 * r_e ** 2 / (2 * reference_orbit.get_a() ** 2) * np.cos(i_sat2)

    q = n * c - (np.sin(gamma_0) * np.cos(gamma_0) * (1 / np.tan(delta_RAAN_0)) - np.sin(gamma_0) ** 2 * np.cos(
        reference_orbit.get_i())) * (d_RAAN_sat1_0 - d_RAAN_sat2_0) - d_RAAN_sat1_0 * np.cos(reference_orbit.get_i())
    l = -reference_orbit.get_a() * (
    np.sin(reference_orbit.get_i()) * np.sin(i_sat2) * np.sin(delta_RAAN_0) / np.sin(phi_0)) * (
        d_RAAN_sat1_0 - d_RAAN_sat2_0)

    m = reference_orbit.get_a() * phi_0
    phi = delta_state_0[2] / m

    return n, c, l, q, phi


def j2_sedwick_propagator(delta_state_0, reference_orbit, time, step, type, thresh_min, thresh_max, target_status):
    n, c, l, q, phi = evaluate_j2_constants(reference_orbit, delta_state_0)
    sc = sp.integrate.ode(lambda t, x: sedwick_eom(t, x, n, c, l, q, phi)).set_integrator('dopri5', atol=1e-5,
                                                                                          rtol=1e-3)
    sc.set_initial_value(delta_state_0, time[0])
    t = np.zeros((len(time), len(delta_state_0)))
    result = np.zeros((len(time), len(delta_state_0)))
    t[0] = time[0]
    result[0][:] = delta_state_0
    step_count = 1

    if type == 0:
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

        return [result[:, 0], result[:, 1], result[:, 2]]

    elif type == 1:  # if we are counting how many fit within a threshold rather than simply propagating

        success_count = 0
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1
            if (np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) > thresh_min) & (
                        np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) < thresh_max):
                success_count += 1

        return success_count

    elif type == 2:
        result_in_range = [[], [], [], [], [], []]
        t_in_range = []
        result_out_of_range = [[], [], [], [], [], []]
        t_out_of_range = []
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            step_count += 1
            if (np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) > thresh_max) | \
                    (np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) < thresh_min):
                for i in range(len(sc.y)):
                    result_out_of_range[i].append(sc.y[i])
                t_out_of_range.append(sc.t)
            else:
                for i in range(len(sc.y)):
                    result_in_range[i].append(sc.y[i])
                t_in_range.append(sc.t)

        return t_in_range, [result_in_range[0], result_in_range[1], result_in_range[2]], \
               t_out_of_range, [result_out_of_range[0], result_out_of_range[1], result_out_of_range[2]]

    elif type == 3:  # to get amounts of time for each pass
        pass_lengths = []
        pass_on = False
        while sc.successful() and step_count < len(t):
            sc.integrate(sc.t + step)
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1
            if (np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) > thresh_min) & (
                        np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2) < thresh_max):
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
            magnitudes.append(np.sqrt(sc.y[0] ** 2 + sc.y[1] ** 2 + sc.y[2] ** 2))

        return magnitudes


def j2_sedwick_targeter(delta_state_0, targeted_state, reference_orbit, time, step, end_seconds, thresh_min,
                        thresh_max, target_status):

    n, c, l, q, phi = evaluate_j2_constants(reference_orbit, delta_state_0)

    # Jacobian matrix
    A = np.zeros(shape=(6, 6))
    A[0][3] = 1
    A[1][4] = 1
    A[2][5] = 1
    A[3][0] = (5 * c ** 2 - 2) * n ** 2  # 24th element
    A[3][4] = 2 * n * c  # 28th element
    A[4][3] = -2 * n * c  # 33rd element
    A[5][2] = -q ** 2  # 38th element

    sc = sp.integrate.ode(lambda t, x: sedwick_eom_st(t, x, A, n, c, l, q, phi)).set_integrator('dopri5', atol=1e-12,
                                                                                          rtol=1e-12)
    sc.set_initial_value(delta_state_0, time[0])

    results = [[], [], [], [], [], []]
    t = []

    d_v = [0.0, 0.0, 0.0]

    stable = True
    current_time = 0
    target_status = True

    while sc.successful() and stable:
        sc.integrate(sc.t + step)
        current_time = current_time + step
        t.append(current_time)
        results[0].append(sc.y[0])
        results[1].append(sc.y[1])
        results[2].append(sc.y[2])
        results[3].append(sc.y[3])
        results[4].append(sc.y[4])
        results[5].append(sc.y[5])
        if sc.t > end_seconds:
        # if np.sqrt((sc.y[0]**2 + sc.y[1]**2 + sc.y[2]**2)) > thresh_max:  # do targeting!
            # determine a maneuver to put the spacecraft back on track :)
            # compute inverse of the state transition matrix
            S_T_inv = np.linalg.inv(TargetingUtils.recompose(sc.y, 6))
            # substitute ideal positions at time = k
            modified_state_time_k = np.asarray([targeted_state[0], targeted_state[1], targeted_state[2], sc.y[3], sc.y[4], sc.y[5]])
            # compute altered state at time = 0
            modified_state_time_0 = np.matmul(S_T_inv, modified_state_time_k)
            # select out the values for the canonical variables we're interested in changing
            d_v = [modified_state_time_0[3] + delta_state_0[3],
                   modified_state_time_0[4] + delta_state_0[4],
                   modified_state_time_0[5] + delta_state_0[5]]

            target_status = False
            stable = False

    return d_v, results, current_time, target_status

# ############################################################################ #
