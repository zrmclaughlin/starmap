import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
import scipy as sp
from scipy import integrate
from sympy.solvers import solve
from sympy import Symbol
from sympy import cos as symcos
from sympy import sin as symsin
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


def sedwick_eom(t, delta_state, n, c, l, q, phi, A):
    state_size = len(delta_state)  # 1: implies x ODEs
    d_delta_state_dt = np.zeros((1, state_size))
    S_T = TargetingUtils.recompose(delta_state, len(A[0]))
    S_T_dt = np.matmul(A, S_T).tolist()

    d_delta_state_dt[0][0] = delta_state[3]
    d_delta_state_dt[0][1] = delta_state[4]
    d_delta_state_dt[0][2] = delta_state[5]
    d_delta_state_dt[0][3] = 2 * n * c * delta_state[4] + (5 * c ** 2 - 2) * n ** 2 * delta_state[0]
    d_delta_state_dt[0][4] = -2 * n * c * delta_state[3]
    d_delta_state_dt[0][5] = -q ** 2 * delta_state[2] + 2 * l * q * np.cos(q * t + phi)

    d_delta_state_dt = np.concatenate(([d_delta_state_dt], S_T_dt), axis=0).flatten()

    return d_delta_state_dt


def j2_sedwick_propagator(delta_state_0, i_sat1, time, step, targeted_state):

    # calculate j2 parameter effects, assuming that the reference
    # orbit is the same as satellite 1's circularized orbit.
    s = 3 * j2 * r_e ** 2 / (8 * a_reference ** 2) * (1 + 3 * np.cos(2 * i_sat1))
    c = np.sqrt(s + 1)
    n = np.sqrt(mu / a_reference ** 3)
    k = n*c + 3*n*j2*r_e**2/(2*a_reference**2)*np.cos(i_sat1)**2

    i_sat2 = i_sat1 - delta_state_0[5]/(k*a_reference)

    delta_RAAN_0 = delta_state_0[2]/(a_reference*np.sin(i_sat1))
    gamma_0 = float(acot( (cot(i_sat2)*np.sin(i_sat1) - np.cos(i_sat1)*np.cos(delta_RAAN_0)) / np.sin(delta_RAAN_0)))
    phi_0 = np.arccos(np.cos(i_sat1)*np.cos(i_sat2) + np.sin(i_sat1)*np.sin(i_sat2)*np.cos(delta_RAAN_0))

    d_RAAN_sat1_0 = -3*n*j2*r_e**2/(2*a_reference**2)*np.cos(i_sat1)
    d_RAAN_sat2_0 = -3*n*j2*r_e**2/(2*a_reference**2)*np.cos(i_sat2)

    q = n*c - (np.sin(gamma_0)*np.cos(gamma_0)*(1/np.tan(delta_RAAN_0)) - np.sin(gamma_0)**2*np.cos(i_sat1))*(d_RAAN_sat1_0 - d_RAAN_sat2_0) - d_RAAN_sat1_0*np.cos(i_sat1)
    l = -a_reference*(np.sin(i_sat1)*np.sin(i_sat2)*np.sin(delta_RAAN_0)/np.sin(phi_0))*(d_RAAN_sat1_0 - d_RAAN_sat2_0)

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
    t = np.zeros((len(time)))
    result = np.zeros((len(time), len(delta_state_0)))
    t[0] = time[0]
    result[0][:] = delta_state_0
    step_count = 1
    target_status = True
    stable = True
    d_v = [0, 0, 0]
    while sc.successful() and stable:
        sc.integrate(sc.t + step)
        # Store the results to plot later
        t[step_count] = sc.t
        result[step_count][:] = sc.y
        step_count += 1
        if step_count > len(t):
        # if np.sqrt((sc.y[0]**2 + sc.y[1]**2 + sc.y[2]**2)) > thresh_max:  # do targeting!
            S_T_inv = np.linalg.inv(TargetingUtils.recompose(sc.y, 6))
            modified_state_time_k = np.asarray([targeted_state[0], targeted_state[1], targeted_state[2], sc.y[3], sc.y[4], sc.y[5]])
            modified_state_time_0 = np.matmul(S_T_inv, modified_state_time_k)
            d_v = [modified_state_time_0[3] + delta_state_0[3],
                   modified_state_time_0[4] + delta_state_0[4],
                   modified_state_time_0[5] + delta_state_0[5]]

            target_status = False
            stable = False

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
            if step_count > len(t)-1 and target:
                S_T_inv = np.linalg.inv(TargetingUtils.recompose(sc.y, 6))
                S_T_rv_vv = S_T_inv[[np.arange(0, 6)[:, None], np.arange(0, 3)[None, :]]]
                equals_v_plus_dv = np.matmul(S_T_rv_vv, np.asarray(targeted_state) - np.asarray(sc.y))
                d_v = [equals_v_plus_dv[3],
                       equals_v_plus_dv[4],
                       equals_v_plus_dv[5]]

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
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    for i in range(7):
        cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, True)
        print("loop", i, cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
        print(d_v)
        delta_state_0 = [10, 100, 10, d_v[0], d_v[1], d_v[2]]
        targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, False)

    print(cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])

    ax = plt.axes(projection='3d')
    ax.set_xlabel("Radial")
    ax.set_ylabel("In-Track")
    ax.set_zlabel("Cross-Track")
    ax.set_title("Relative Motion")

    # Data for a three-dimensional line
    ax.plot3D(cw_results[-1][0], cw_results[-1][1], cw_results[-1][2])
    plt.show()


def test_stm(delta_state_0, times, step, nominal_position):
    targeted_state = np.concatenate(([delta_state_0], np.eye(len(delta_state_0))), axis=0).flatten()

    cw_t, cw_results, target_status, stable, d_v = cw_propagator(times, targeted_state, step, nominal_position, False)
    last_stm = TargetingUtils.recompose(flat_state=cw_results[-1], state_length=6)
    final_truth = np.asarray(cw_results[-1][:6])
    final_guess = np.matmul(last_stm, np.asarray(delta_state_0))

    print("Time: ", cw_t[-1], ": Results:", final_truth)
    print("Time: ", cw_t[-1], ": Results:", final_guess)
    print("Time: 1000:", cw_stm(delta_state_0, 1000))

    # test passes as of 11/18/2019

    return


def main():
    delta_state_0 = [10, 100, 10, 1, 2, 3]
    times = np.linspace(0, 1000, 1000)
    inc_reference = 30 * np.pi / 180
    step = times[1] - times[0]
    nominal_position = [1000, 1000, 100]

    # test_targeter(delta_state_0, times, step, nominal_position)
    test_stm(delta_state_0, times, step, nominal_position)

    return


if __name__ == "__main__":
    main()
