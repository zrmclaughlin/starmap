import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
import scipy as sp
from scipy import integrate
from sympy import *
from mpmath import *
import OrbitalElements
import TargetingUtils
import CombinedModelJacobian as jcb

# Where:
# radial = results[:, 0]
# in_track = results[:, 1]
# z = results[:, 2]

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
a_reference = 6378136.3 + 300000
inc_reference = 30 * np.pi / 180


def sedwick_eom(t, delta_state, n, c, l, q, phi, A):
    d_delta_state_dt = np.zeros(shape=(1, 6))
    S_T = TargetingUtils.recompose(delta_state, 6)
    S_T_dt = np.matmul(A, S_T).tolist()

    d_delta_state_dt[0][0] = delta_state[3]
    d_delta_state_dt[0][1] = delta_state[4]
    d_delta_state_dt[0][2] = delta_state[5]
    d_delta_state_dt[0][3] = 2 * n * c * delta_state[4] + (5 * c ** 2 - 2) * n ** 2 * delta_state[0]
    d_delta_state_dt[0][4] = -2 * n * c * delta_state[3]
    d_delta_state_dt[0][5] = -q ** 2 * delta_state[2] + 2 * l * q * np.cos(q * t + phi)

    # print(S_T_dt.size)

    d_delta_state_dt = np.concatenate((d_delta_state_dt, S_T_dt), axis=0).flatten()

    return d_delta_state_dt


def j2_sedwick_propagator(time, delta_state_0, step, targeted_state, target):

    # calculate j2 parameter effects, assuming that the reference
    # orbit is the same as satellite 1's circularized orbit.
    s = 3 * j2 * r_e ** 2 / (8 * a_reference ** 2) * (1 + 3 * np.cos(2 * inc_reference))
    c = np.sqrt(s + 1)
    n = np.sqrt(mu / a_reference ** 3)
    k = n*c + 3*n*j2*r_e**2/(2*a_reference**2)*np.cos(inc_reference)**2

    i_sat2 = float(inc_reference - delta_state_0[5]/(k*a_reference))

    delta_RAAN_0 = delta_state_0[2]/(a_reference*np.sin(inc_reference))
    gamma_0 = float(acot((cot(i_sat2)*np.sin(inc_reference) - np.cos(inc_reference)*np.cos(delta_RAAN_0)) / np.sin(delta_RAAN_0)))
    phi_0 = np.arccos(np.cos(inc_reference)*np.cos(i_sat2) + np.sin(inc_reference)*np.sin(i_sat2)*np.cos(delta_RAAN_0))

    d_RAAN_sat1_0 = -3*n*j2*r_e**2/(2*a_reference**2)*np.cos(inc_reference)
    d_RAAN_sat2_0 = -3*n*j2*r_e**2/(2*a_reference**2)*np.cos(i_sat2)

    q = n*c - (np.sin(gamma_0)*np.cos(gamma_0)*(1/np.tan(delta_RAAN_0)) - np.sin(gamma_0)**2*np.cos(inc_reference))*(d_RAAN_sat1_0 - d_RAAN_sat2_0) - d_RAAN_sat1_0*np.cos(inc_reference)
    l = -a_reference*(np.sin(inc_reference)*np.sin(i_sat2)*np.sin(delta_RAAN_0)/np.sin(phi_0))*(d_RAAN_sat1_0 - d_RAAN_sat2_0)

    m = a_reference * phi_0
    phi = delta_state_0[2] / m

    # Jacobian matrix
    A = np.zeros(shape=(6, 6))
    A[0][3] = 1
    A[1][4] = 1
    A[2][5] = 1
    A[3][0] = (5 * c ** 2 - 2) * n ** 2  # 24th element
    A[3][4] = 2 * n * c  # 28th element
    A[4][3] = -2 * n * c  # 33rd element
    A[5][2] = -q ** 2  # 38th element

    sc = sp.integrate.ode(lambda t, x: sedwick_eom(t, x, n, c, l, q, phi, A)).set_integrator('dopri5', atol=1e-12, rtol=1e-12)
    sc.set_initial_value(delta_state_0, time[0])

    t = np.zeros((len(time)+1))
    result = np.zeros((len(time)+1, len(delta_state_0)))
    t[0] = time[0]
    result[0][:] = delta_state_0

    step_count = 1
    target_status = True
    stable = True
    d_v = [0, 0, 0]

    if target:
        while sc.successful() and stable:
            sc.integrate(sc.t + step)
            # Store the results to plot later
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

            if step_count > len(t) - 1 and target:
                S_T = TargetingUtils.recompose(sc.y, 6)
                S_T_rv_vv = Matrix(S_T[np.arange(0, 6)[:, None], np.arange(3, 6)[None, :]])
                initial_d_dv1, initial_d_dv2, initial_d_dv3 = symbols('initial_d_dv1 initial_d_dv2 initial_d_dv3',
                                                                      real=True)
                initial_d_dv = Matrix([initial_d_dv1, initial_d_dv2, initial_d_dv3])
                S_T_times_initial_d_dv = S_T_rv_vv*initial_d_dv
                final_d_dp = np.asarray(targeted_state) - sc.y[:3]

                eqs = [S_T_times_initial_d_dv[0] - final_d_dp[0],
                       S_T_times_initial_d_dv[1] - final_d_dp[1],
                       S_T_times_initial_d_dv[2] - final_d_dp[2]]

                reeeeee = linsolve(eqs, initial_d_dv1, initial_d_dv2, initial_d_dv3).args[0]
                d_v = [reeeeee[0], reeeeee[1], reeeeee[2]]  # reference - actual

                target_status = False
                stable = False

            elif step_count > len(t)-1 and not target:
                stable = False

    elif not target:
        while sc.successful() and stable and step_count < len(t):
            sc.integrate(sc.t + step)
            # Store the results to plot later
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

    return t, result, target_status, stable, d_v


def cw_eom(t, delta_state, a_target, mu, A):  # semi-major of target satellite
    n = np.sqrt(mu/a_target**3)
    S_T = TargetingUtils.recompose(delta_state, 6)
    S_T_dt = np.matmul(A(delta_state[0], delta_state[1], delta_state[2], delta_state[3], delta_state[4], delta_state[5]), S_T)
    d_delta_state_dt = np.zeros((1, 6))
    d_delta_state_dt[0][0] = delta_state[3]
    d_delta_state_dt[0][1] = delta_state[4]
    d_delta_state_dt[0][2] = delta_state[5]
    d_delta_state_dt[0][3] = 3*n**2*delta_state[0] + 2*n*delta_state[4]
    d_delta_state_dt[0][4] = - 2*n*delta_state[3]
    d_delta_state_dt[0][5] = - n**2*delta_state[2]
    d_delta_state_dt = np.concatenate((d_delta_state_dt, S_T_dt), axis=0).flatten()
    return d_delta_state_dt


def cw_stm(x0, t):  # semi-major of target satellite
    n = np.sqrt(mu/a_reference**3)
    cw = [[4-3*np.cos(n*t),       0,  0,              1/n*np.sin(n*t),     2/n*(1-np.cos(n*t)),         0],
            [6*(np.sin(n*t) - n*t), 1,  0,             -2/n*(1-np.cos(n*t)), 1/n*(4*np.sin(n*t) - 3*n*t), 0],
            [0,                     0,  np.cos(n*t),    0,                   0,                           1/n*np.sin(n*t)],
            [3*n*np.sin(n*t),       0,  0,              np.cos(n*t),          2*np.sin(n*t),               0],
            [-6*n*(1-np.cos(n*t)),  0,  0,             -2*np.sin(n*t),       4*np.cos(n*t) - 3,           0],
            [0,                     0, -n*np.sin(n*t),  0,                   0,                           np.cos(n*t)]]
    return np.matmul(cw, x0)


def cw_propagator(time, delta_state_0, step, targeted_state, target):
    A = jcb.first_order_jacobian(a_reference)

    sc = sp.integrate.ode(lambda t, x: cw_eom(t, x, a_target=a_reference, mu=mu, A=A)).set_integrator('dopri5', atol=1e-12, rtol=1e-12)
    sc.set_initial_value(delta_state_0, time[0])

    t = np.zeros((len(time)+1))
    result = np.zeros((len(time)+1, len(delta_state_0)))
    t[0] = time[0]
    result[0][:] = delta_state_0

    step_count = 1
    target_status = True
    stable = True
    d_v = [0, 0, 0]

    if target:
        while sc.successful() and stable:
            sc.integrate(sc.t + step)
            # Store the results to plot later
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

            if step_count > len(t) - 1 and target:
                S_T = TargetingUtils.recompose(sc.y, 6)
                S_T_rv_vv = Matrix(S_T[np.arange(0, 6)[:, None], np.arange(3, 6)[None, :]])
                initial_d_dv1, initial_d_dv2, initial_d_dv3 = symbols('initial_d_dv1 initial_d_dv2 initial_d_dv3',
                                                                      real=True)
                initial_d_dv = Matrix([initial_d_dv1, initial_d_dv2, initial_d_dv3])
                S_T_times_initial_d_dv = S_T_rv_vv*initial_d_dv
                final_d_dp = np.asarray(targeted_state) - sc.y[:3]

                eqs = [S_T_times_initial_d_dv[0] - final_d_dp[0],
                       S_T_times_initial_d_dv[1] - final_d_dp[1],
                       S_T_times_initial_d_dv[2] - final_d_dp[2]]

                reeeeee = linsolve(eqs, initial_d_dv1, initial_d_dv2, initial_d_dv3).args[0]
                d_v = [reeeeee[0], reeeeee[1], reeeeee[2]]  # reference - actual

                target_status = False
                stable = False

            elif step_count > len(t)-1 and not target:
                stable = False

    elif not target:
        while sc.successful() and stable and step_count < len(t):
            sc.integrate(sc.t + step)
            # Store the results to plot later
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

    return t, result, target_status, stable, d_v


def test_targeter(delta_state_0, times, step, nominal_position):
    delta_state_cw = [delta_state_0[0], delta_state_0[1], delta_state_0[2], delta_state_0[3], delta_state_0[4], delta_state_0[5]]
    delta_state_J2 = [delta_state_0[0], delta_state_0[1], delta_state_0[2], delta_state_0[3], delta_state_0[4], delta_state_0[5]]

    # Test 1 - CW
    print("++++++++ CW TARGETING TEST ++++++++")
    targeted_state = np.concatenate(([delta_state_cw], np.eye(len(delta_state_cw))), axis=0).flatten()
    final_state = [10000, 10000, 10000]
    counter = 0
    while ((np.linalg.norm(np.asarray(final_state)) - np.linalg.norm(np.asarray(nominal_position))) > 10) and (counter < 10):
        cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, True)
        print("Loop", counter, " | Final Position:", cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
        final_state = [cw_results[-1][0], cw_results[-1][1], cw_results[-1][2]]
        print("|->  Delta delta V: ", d_v)
        delta_state_cw = [delta_state_cw[0], delta_state_cw[1], delta_state_cw[2],
                         delta_state_cw[3] + d_v[0], delta_state_cw[4] + d_v[1], delta_state_cw[5] + d_v[2]]
        print("|->  New Relative State: ", delta_state_cw)
        targeted_state = np.concatenate(([delta_state_cw], np.eye(len(delta_state_cw))), axis=0).flatten()
        counter = counter + 1

    cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, False)
    print("Post-Targeting State: ", cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
    print("Done. Total Loops: ", counter)

    # test passed as of 11/19/2019

    # Test 2 - J2
    print("++++++++ J2 TARGETING TEST ++++++++")
    targeted_state = np.concatenate(([delta_state_J2], np.eye(len(delta_state_J2))), axis=0).flatten()
    final_state = [10000, 10000, 10000]
    counter = 0
    while ((np.linalg.norm(np.asarray(final_state)) - np.linalg.norm(np.asarray(nominal_position))) > 10) and (counter < 10):
        j2_t, j2_results, target_status, stable, d_v = j2_sedwick_propagator(times, targeted_state, step, nominal_position, True)
        print("Loop", counter, " | Final Position:", j2_results[-1][0], j2_results[-1][1], j2_results[-1][2])
        final_state = [j2_results[-1][0], j2_results[-1][1], j2_results[-1][2]]
        print("|->  Delta delta V: ", d_v)
        delta_state_J2 = [delta_state_J2[0], delta_state_J2[1], delta_state_J2[2],
                         delta_state_J2[3] + d_v[0], delta_state_J2[4] + d_v[1], delta_state_J2[5] + d_v[2]]
        print("|->  New Relative State: ", delta_state_J2)
        targeted_state = np.concatenate(([delta_state_J2], np.eye(len(delta_state_J2))), axis=0).flatten()
        counter = counter + 1

    j2_t, j2_results, target_status, stable, d_v = j2_sedwick_propagator(times, targeted_state, step, nominal_position, False)
    print("Post-Targeting State: ", j2_results[-1][0], j2_results[-1][1], j2_results[-1][2])
    print("Done. Total Loops: ", counter)


def test_stm(delta_state_0, times, step, nominal_position):
    print("Time: 1000:", "CW Analytic STM", cw_stm(delta_state_0, 1000))
    # Test 1 - CW
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, False)
    last_stm = TargetingUtils.recompose(flat_state=cw_results[-1], state_length=6)
    final_truth = np.asarray(cw_results[-1][:6])
    final_guess = np.matmul(last_stm, np.asarray(delta_state_0))

    print("Time: ", cw_t[-1], ": Results CW:", final_truth)
    print("Time: ", cw_t[-1], ": Results STM CW:", final_guess)
    # test passes as of 11/18/2019

    # Test 2 - J2
    j2_t, j2_results, target_status, stable, d_v = j2_sedwick_propagator(times, targeted_state, step, nominal_position, False)
    j2_last_stm = TargetingUtils.recompose(j2_results[-1], state_length=6)
    j2_final_truth = np.asarray(j2_results[-1][:6])
    j2_final_guess = np.matmul(j2_last_stm, delta_state_0)

    print("Time: ", j2_t[-1], ": Results J2:", j2_final_truth)
    print("Time: ", j2_t[-1], ": Results STM J2:", j2_final_guess)
    # test passes as of 11/18/2019


def main():
    delta_state_0 = [10, 10, 10, 1, 2, 3]
    times = np.linspace(0, 100, 100)
    step = times[1] - times[0]
    nominal_position = [100, 1000, 20]

    test_targeter(delta_state_0, times, step, nominal_position)
    # test_stm(delta_state_0, times, step, nominal_position)

    return


if __name__ == "__main__":
    main()
