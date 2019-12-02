from sympy import *
from sympy import exp
import numpy as np

r_e = 6378136.3
j2 = 1.082E-3
mu = 3.986004415E14
k_j2 = 3*j2*mu*r_e**2 / 2


def get_jacobian(c_d, a_m_reference, a_m_chaser, r_0, rho_0, H):
    # State Variables for combination
    r_reference, v_z, h_reference, theta_reference, i_reference, x, y, z, p1, p2, p3 = \
        symbols('r_reference v_z h_reference theta_reference i_reference x y z p1 p2 p3', real=True)

    wy = -h_reference / r_reference**2
    wz = k_j2 * sin(2*i_reference) * sin(theta_reference) / (h_reference * r_reference**3)

    r_chaser = Matrix([x, y, z - r_reference]).norm()
    z_chaser = x*cos(theta_reference)*sin(i_reference) - y*cos(i_reference) - (z - r_reference)*sin(i_reference)*sin(theta_reference)
    w_bar = -mu / r_chaser**3 - k_j2 / r_chaser**5 + 5*k_j2*z_chaser**2 / r_chaser**7
    zeta = 2*k_j2*z_chaser / r_chaser**5

    v_reference = Matrix([h_reference/r_reference, 0, v_z]).norm()
    rho_reference = rho_0*exp(-(r_reference - r_0)/H)
    f_drag_reference = - .5*c_d*a_m_reference*rho_reference

    dr_dt = -v_z  # d r / dt
    dv_z_dt = mu / r_reference**2 - h_reference**2 / r_reference**3 + k_j2*(1 - 3*sin(i_reference)**2 * sin(theta_reference)**2) / r_reference**4 + f_drag_reference*v_z*v_reference  # d v_z / dt
    dh_reference_dt = -k_j2*sin(i_reference)**2*sin(2*theta_reference) / r_reference**3 + f_drag_reference*h_reference*v_reference  # d h_reference / dt
    dtheta_reference_dt = h_reference / r_reference**2 + 2*k_j2*cos(i_reference)**2*sin(theta_reference)**2 / (h_reference * r_reference**3)  # d theta_reference / dt
    di_reference_dt = -k_j2*sin(2*theta_reference)*sin(2*i_reference) / (2 * h_reference * r_reference**3)  # d i_reference / dt
    dx_dt = p1 + y*wz - (z - r_reference)*wy  # d x / dt
    dy_dt = p2 - x*wz  # d y / dt
    dz_dt = p3 - v_z + x*wy  # d z / dt

    v_chaser = Matrix([dx_dt - y*wz + (z - r_reference)*wy, dy_dt + x*wz, dz_dt + v_z - x*wy])
    rho_chaser = rho_0*exp(-(r_chaser - r_0)/H)
    v_chaser_n = v_chaser.norm()
    f_drag_chaser = - .5*c_d*a_m_chaser*rho_chaser*v_chaser_n

    dp1_dt = w_bar*x - zeta*cos(theta_reference)*sin(i_reference) + p2*wz - p3*wy + f_drag_chaser*v_chaser[0]  # d p1 / dt
    dp2_dt = w_bar*y + zeta*cos(i_reference) - p1*wz + f_drag_chaser*v_chaser[1]  # d p2 / dt
    dp3_dt = w_bar*(z - r_reference) + zeta*sin(theta_reference)*sin(i_reference) + p1*wy + f_drag_chaser*v_chaser[2]  # d p3 / dt

    CombinedJacobian = Matrix([dr_dt, dv_z_dt, dh_reference_dt, dtheta_reference_dt, di_reference_dt, dx_dt, dy_dt, dz_dt, dp1_dt, dp2_dt, dp3_dt]).\
                        jacobian([r_reference, v_z, h_reference, theta_reference, i_reference, x, y, z, p1, p2, p3])

    return lambdify((r_reference, v_z, h_reference, theta_reference, i_reference, x, y, z, p1, p2, p3),
                   CombinedJacobian, modules='sympy')


def first_order_jacobian(a_target):
    # state variables for first order
    x, y, z, dx_dt, dy_dt, dz_dt, d_dx_dt_dt, d_dy_dt_dt, d_dz_dt_dt, d_x_dt, d_y_dt, d_z_dt = \
        symbols('x, y, z, dx_dt dy_dt dz_dt d_dx_dt_dt d_dy_dt_dt d_dz_dt_dt d_x_dt d_y_dt d_z_dt', real=True)
    n = np.sqrt(mu/a_target**3)
    d_x_dt = dx_dt
    d_y_dt = dy_dt
    d_z_dt = dz_dt
    d_dx_dt_dt = 3*n**2*x + 2*n*dy_dt
    d_dy_dt_dt = - 2*n*dx_dt
    d_dz_dt_dt = - n**2*z

    FirstOrderJacobian = Matrix([d_x_dt, d_y_dt, d_z_dt, d_dx_dt_dt, d_dy_dt_dt, d_dz_dt_dt]).\
                        jacobian([x, y, z, dx_dt, dy_dt, dz_dt])

    return lambdify((x, y, z, dx_dt, dy_dt, dz_dt),
                   FirstOrderJacobian, modules='sympy')
