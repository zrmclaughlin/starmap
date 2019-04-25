import numpy as np
import scipy as sp
from scipy import integrate
from mpmath import *

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14


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


def j2_sedwick_propagator(delta_state_0, reference_orbit, time, step, type, thresh_min, thresh_max):
    print("min:", thresh_min)
    print("max: ", thresh_max)
    print(delta_state_0)
    n, c, l, q, phi = evaluate_j2_constants(reference_orbit, delta_state_0)
    sc = sp.integrate.ode(lambda t, x: sedwick_eom(t, x, n, c, l, q, phi)).set_integrator('dopri5', atol=1e-12,
                                                                                          rtol=1e-12)
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
            # result[step_count][:] = sc.y
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

# ############################################################################ #
