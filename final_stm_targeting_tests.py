import numpy as np
import scipy as sp
from scipy import integrate

import CombinedModelJacobian as CJJacobian
import TargetingUtils

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2


def combined_model_eom(t, state, A, c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):

    # <- r_reference, v_z, h_reference, theta_reference, i_reference, x_0, y_0, z_0, p1, p2, p3
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

    return dstate_dt


def combined_targeter():
    return


def main():
    return


if __name__ == "__main__":
    main()