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
# print(k_j2/r_e**3)

def velocity_from_state(wy, wz, r_reference, v_z, x, y, z, p1, p2, p3):
    vx = p1 + y*wz - (z - r_reference)*wy
    vy = p2 - x*wz
    vz = p3 - v_z + x*wy
    return [vx, vy, vz]


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
    # v_chaser = [state[8], state[9], state[10]]
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
                S_T_rv_vv = Matrix(S_T[np.arange(0, 11)[:, None], np.arange(3, 11)[None, :]])
                initial_d_dv1, initial_d_dv2, initial_d_dv3 = symbols('initial_d_dv1 initial_d_dv2 initial_d_dv3',
                                                                      real=True)
                initial_d_dv = Matrix([initial_d_dv1, initial_d_dv2, initial_d_dv3])
                S_T_times_initial_d_dv = S_T_rv_vv * initial_d_dv
                final_d_dp = np.asarray(targeted_state) - sc.y[:3]

                eqs = [S_T_times_initial_d_dv[0] - final_d_dp[0],
                       S_T_times_initial_d_dv[1] - final_d_dp[1],
                       S_T_times_initial_d_dv[2] - final_d_dp[2]]

                reeeeee = linsolve(eqs, initial_d_dv1, initial_d_dv2, initial_d_dv3).args[0]
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
    # 1st order reference
    print("Time: ", times[-1], ":", "CW Analytic STM", targeting_test.cw_stm(six_state, times[-1]))

    # Combined motion test
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    cw_t, cw_results, target_status, stable, d_v = combined_targeter(times, targeted_state, step, nominal_position, False, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)
    last_differential_stm = TargetingUtils.recompose(flat_state=cw_results[-1], state_length=11)

    # Now that I have the final differential STM, I need to propagate a state modified by a differential and find the corresponding
    # final modified state! I can then compare this to the STM
    differential = [1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]
    differential = [i*random.random()*10 for i in differential]
    print("Initial State Differential: ", differential)



    final_truth = np.asarray(cw_results[-1][:11])
    final_guess = np.matmul(last_differential_stm, np.asarray(delta_state_0))

    print("Time: ", cw_t[-1], ": Results Combined:", final_truth)
    print("Time: ", cw_t[-1], ": Results STM Combined:", final_guess)
    # test fails




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

    times = np.linspace(0, 10, 10)
    step = times[1] - times[0]
    nominal_position = [100, 1000, 20]

    # The xr axis completes the right-handed frame -> transverse (y)
    # The yr axis points to the opposite direction of the orbital angular moment - > negative normal (-z)
    # The zr axis points to the center of the Earth -> negative radial (-x)
    cw_frame_state = [-z_0, x_0, -y_0, -z_0_dot, x_0_dot, -y_0_dot]

    test_stm(cw_frame_state, delta_state_0, times, step, nominal_position, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H)

    return


if __name__ == "__main__":
    main()
