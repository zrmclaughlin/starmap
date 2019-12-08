import numpy as np
import scipy as sp
from scipy import integrate
from sympy import *
import CombinedModelJacobian as CJJacobian
import TargetingUtils
import targeting_test
import random

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2


def combined_model_eom(t, state, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3
    wy = -state[2] / state[0]**2
    wz = k_j2 * np.sin(2*state[4]) * np.sin(state[3]) / (state[2] * state[0]**3)

    r_chaser = np.linalg.norm([state[5], state[6], state[7] - state[0]])
    z_chaser = state[5]*np.cos(state[3])*np.sin(state[4]) - state[6]*np.cos(state[4]) - (state[7] - state[0])*np.sin(state[4])*np.sin(state[3])

    w_bar = -mu / r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*z_chaser**2 / r_chaser**7
    zeta = 2*k_j2*z_chaser / r_chaser**5

    v_reference = np.linalg.norm([state[2]/state[0], 0, state[1]])
    rho_reference = rho_0*exp(-(state[0] - r_0)/H)
    f_drag_reference = - .5*c_d*a_m_reference*rho_reference

    dstate_dt = np.zeros(shape=(1, 11))

    dstate_dt[0][0] = -state[1]  # d r / dt
    dstate_dt[0][1] = mu / state[0]**2 - state[2]**2 / state[0]**3 + k_j2*(1 - 3*np.sin(state[4])**2 * np.sin(state[3])**2) / state[0]**4 + f_drag_reference*state[1]*v_reference  # d v_z / dt
    dstate_dt[0][2] = -k_j2*np.sin(state[4])**2*np.sin(2*state[3]) / state[0]**3 + f_drag_reference*state[2]*v_reference # d h_reference / dt
    dstate_dt[0][3] = state[2] / state[0]**2 + 2*k_j2*np.cos(state[4])**2*np.sin(state[3])**2 / (state[2] * state[0]**3)  # d theta_reference / dt
    dstate_dt[0][4] = -k_j2*np.sin(2*state[3])*np.sin(2*state[4]) / (2 * state[2] * state[0]**3)  # d i_reference / dt
    dstate_dt[0][5] = state[8] + state[6]*wz - (state[7] - state[0])*wy  # d x / dt
    dstate_dt[0][6] = state[9] - state[5]*wz  # d y / dt
    dstate_dt[0][7] = state[10] - state[1] + state[5]*wy  # d z / dt

    v_chaser = [dstate_dt[0][5] - state[6]*wz + (state[7] - state[0])*wy, dstate_dt[0][6] + state[5]*wz, dstate_dt[0][7] + state[1] - state[5]*wy]
    rho_chaser = rho_0*exp(-(r_chaser - r_0)/H)
    f_drag_chaser = - .5*c_d*a_m_chaser*rho_chaser*np.linalg.norm(v_chaser)

    dstate_dt[0][8] = w_bar*state[5] - zeta*np.cos(state[3])*np.sin(state[4]) + state[9]*wz - state[10]*wy + f_drag_chaser*v_chaser[0]  # d p1 / dt
    dstate_dt[0][9] = w_bar*state[6] + zeta*np.cos(state[4]) - state[8]*wz + f_drag_chaser*v_chaser[1]  # d p2 / dt
    dstate_dt[0][10] = w_bar*(state[7] - state[0]) + zeta*np.sin(state[3])*np.sin(state[4]) + state[8]*wy + f_drag_chaser*v_chaser[2]  # d p3 / dt

    S_T = TargetingUtils.recompose(state, 11)
    S_T_dt = np.matmul(A(state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10]), S_T)

    dstate_dt = np.concatenate((dstate_dt, S_T_dt), axis=0).flatten()

    return dstate_dt


def combined_targeter(time, delta_state_0, step, targeted_state, target, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # Jacobian matrix
    A = CJJacobian.get_jacobian(c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)

    sc = sp.integrate.ode(
        lambda t, x: combined_model_eom(t, x, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)).set_integrator('dopri5',
                                                                                                        atol=1e-10,
                                                                                                        rtol=1e-5)
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
                S_T = TargetingUtils.recompose(sc.y, 11)
                # Slice matrix - we only care about the elements used to target position
                S_T_xyz_p123 = Matrix(S_T[np.arange(5, 8)[:, None], np.arange(8, 11)[None, :]])

                initial_d_dp1, initial_d_dp2, initial_d_dp3 = symbols('initial_d_dp1 initial_d_dp2 initial_d_dp3',
                                                                      real=True)

                initial_d_dv = Matrix([initial_d_dp1, initial_d_dp2, initial_d_dp3])
                S_T_times_initial_d_dp = S_T_xyz_p123 * initial_d_dv

                # Delta relative position
                final_d_dp = np.asarray(targeted_state) - sc.y[5:8]

                eqs = [S_T_times_initial_d_dp[0] - final_d_dp[0],
                       S_T_times_initial_d_dp[1] - final_d_dp[1],
                       S_T_times_initial_d_dp[2] - final_d_dp[2]]

                reeeeee = linsolve(eqs, initial_d_dp1, initial_d_dp2, initial_d_dp3).args[0]
                print(eqs)
                print(reeeeee)
                d_v = [reeeeee[0], reeeeee[1], reeeeee[2]]  # reference - actual

                target_status = False
                stable = False

            elif step_count > len(t) - 1 and not target:
                stable = False

    elif not target:
        while sc.successful() and stable and step_count < len(t):
            sc.integrate(sc.t + step)
            # Store the results to plot later
            t[step_count] = sc.t
            result[step_count][:] = sc.y
            step_count += 1

    return t, result, target_status, stable, d_v


def test_stm(six_state, delta_state_0, times, step, nominal_position, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # Combined motion test
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    cw_t, cw_results, target_status, stable, d_v = \
        combined_targeter(times, targeted_state, step, nominal_position,
                          False, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)

    last_differential_stm = TargetingUtils.recompose(flat_state=cw_results[-1], state_length=11)

    print("ROUGH FINAL STATE ACCURACY CHECK \n")
    # 1st order reference final state computation
    print("Time: ", cw_t[-1], ":", "Final State Through CW Analytic STM", targeting_test.cw_stm(six_state, cw_t[-1]))

    # Combined model final state computation
    print("Time: ", cw_t[-1], ":", "Final State Through Combined Final State Propagation:", cw_results[-1][:11], "\n")
    print("================================\n\n")

    print("DIFFERENTIAL STATE TRANSITION MATRIX CHECK")
    # Now that I have the final differential STM, I need to propagate a
    # state modified by a differential and find the corresponding
    # final modified state! I can then compare this to the STM
    differential = [1e2, 1e-10, 1e4, 1e-7, 1e-7, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]
    differential = [i * random.uniform(-1, 1) * 10 for i in differential]
    print("Initial State Differential: ", differential, "\n")

    # Now, propagate modified state through numeric system
    delta_state_0_with_differential = [delta_state_0[i] + differential[i] for i in range(11)]
    targeted_state = np.concatenate(([delta_state_0_with_differential], np.eye(len(delta_state_0))), axis=0).flatten()
    cw_t_mod, cw_results_mod, target_status_mod, stable_mod, d_v_mod = \
        combined_targeter(times, targeted_state, step, nominal_position,
                          False, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)

    xf = cw_results[-1][:11]
    xf_with_differential = cw_results_mod[-1][:11]
    difference_in_final_states = [xf_with_differential[i] - xf[i] for i in range(len(xf))]
    print("Difference in final states using propagator: ", difference_in_final_states, "\n")

    # Now, calculate differences
    difference_from_dstm = np.matmul(last_differential_stm, differential)
    print("Difference in final states using STM",
          difference_from_dstm, "\n")

    print("Ratio Between Differentials",
          np.linalg.norm(difference_in_final_states)/np.linalg.norm(difference_from_dstm))

    print("Absolute Norm Difference Between Differentials",
          np.linalg.norm(difference_in_final_states) - np.linalg.norm(difference_from_dstm), "\n")

    print("================================\n")

    # test passes as of 12/02/2019... omfg


def test_targeter(six_state, nominal_six_position, delta_state_0, times, step, nominal_position, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # Test 1 - CW
    print("++++++++ CW TARGETING TEST ++++++++")
    targeted_state = np.concatenate(([six_state], np.eye(len(six_state))), axis=0).flatten()
    final_state = [10000, 10000, 10000]
    counter = 0
    while (np.abs((np.linalg.norm(np.asarray(final_state)) - np.linalg.norm(np.asarray(nominal_six_position)))) > 10) and (counter < 10):
        cw_t, cw_results, target_status, stable, d_v = targeting_test.cw_propagator(times, targeted_state, step, nominal_six_position, True)
        print("Loop", counter, " | Final Position:", cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
        final_state = [cw_results[-1][0], cw_results[-1][1], cw_results[-1][2]]
        print("|->  Delta delta V: ", d_v)
        six_state = [six_state[0], six_state[1], six_state[2],
                         six_state[3] + d_v[0], six_state[4] + d_v[1], six_state[5] + d_v[2]]
        print("|->  New Relative State: ", six_state)
        targeted_state = np.concatenate(([six_state], np.eye(len(six_state))), axis=0).flatten()
        counter = counter + 1

    cw_t, cw_results, target_status, stable, d_v = targeting_test.cw_propagator(times, targeted_state, step, nominal_six_position, False)
    print("Post-Targeting State: ", cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
    print("Done. Total Loops: ", counter, "\n")

    print("================================\n")

    # test passed as of 11/19/2019

    # Test 2 - Combinted model
    print("++++++++ COMBINED TARGETING TEST ++++++++")
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()
    final_state = [10000, 10000, 10000]
    counter = 0
    while (np.abs((np.linalg.norm(np.asarray(final_state)) - np.linalg.norm(np.asarray(nominal_position)))) > 10) and (counter < 10):
        combined_t, combined_results, target_status, stable, d_v = \
            combined_targeter(times, targeted_state, step, nominal_position,
                              True, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
        print("Loop", counter, " | Final Position:", combined_results[-1][5], combined_results[-1][6], combined_results[-1][7])
        final_state = [combined_results[-1][5], combined_results[-1][6], combined_results[-1][7]]
        print("|->  Delta delta V: ", d_v)
        delta_state_0 = [delta_state_0[0], delta_state_0[1], delta_state_0[2], delta_state_0[3],
                          delta_state_0[4], delta_state_0[5], delta_state_0[6], delta_state_0[7],
                          delta_state_0[8] + d_v[0], delta_state_0[9] + d_v[1], delta_state_0[10] + d_v[2]]
        print("|->  New Relative State: ", delta_state_0)
        targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()
        counter = counter + 1

    combined_t, combined_results, target_status, stable, d_v = \
        combined_targeter(times, targeted_state, step, nominal_position,
                          False, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    print("Post-Targeting State: ", combined_results[-1][5], combined_results[-1][6], combined_results[-1][7])
    print("Done. Total Loops: ", counter, "\n")

    print("================================\n")

    # test passed as of 12/02/2019


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
    y_0 = 5671
    y_0_dot = -13.902
    z_0 = 85
    z_0_dot = -1.993

    wy = -h_reference / r_reference**2
    wz = k_j2 * np.sin(2*i_reference) * np.sin(theta_reference) / (h_reference * r_reference**3)

    p1 = x_0_dot - y_0*wz + (z_0 - r_reference)*wy
    p2 = y_0_dot + x_0*wz
    p3 = z_0_dot + v_z - x_0*wy

    delta_state_0 = [r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3]

    times = np.linspace(0, 100, 50)
    step = times[1] - times[0]
    nominal_position = [40000, 6000, 150]
    nominal_cw_position = [-150, 40000, -6000]

    # The xr axis completes the right-handed frame -> transverse (y)
    # The yr axis points to the opposite direction of the orbital angular moment - > negative normal (-z)
    # The zr axis points to the center of the Earth -> negative radial (-x)
    cw_frame_state = [-z_0, x_0, -y_0, -z_0_dot, x_0_dot, -y_0_dot]
    test_stm(cw_frame_state, delta_state_0, times, step, nominal_position, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    test_targeter(cw_frame_state, nominal_cw_position, delta_state_0, times, step, nominal_position, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)


if __name__ == "__main__":
    main()
