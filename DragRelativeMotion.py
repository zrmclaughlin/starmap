import numpy as np
import OrbitalElements


# ############################## CARTER-HUMI DRAG REOM ############################## #

def st_drag_carter_humi(alpha, reference_orbit, mu, x0, t):  # semi-major of target satellite
    k = np.sqrt(1 - 12*alpha**2)
    n = np.sqrt(mu/reference_orbit.get_a()**3)
    cw = [[1,     4*np.sin(k*t*n)/k**2 + (1 - 4/k**2)*t*n,  2*(1-4/k**2)*(np.sin(k*t*n)/k - t*n),  2*(1 - np.cos(n*t))/k, 0,            0],
          [0,     4*(np.cos(k*t*n) - 1)/k**2 + 1,           2*(1-4/k**2)*(np.cos(k*t*n) - 1),      2*np.sin(k*n*t)/k,     0,            0],
          [0,     2*(np.cos(k*t*n) - 1)/k**2,              (1 - 4/k**2)*np.cos(k*t*n) + 4/k**2,    np.sin(k*t*n)/k,       0,            0],
          [0,     0,                                        0,                                     np.cos(k*n*t),         0,            0],
          [0,     0,                                        0,                                     0,                     np.cos(n*t),  np.sin(t*n)],
          [0,     0,                                        0,                                     0,                    -np.sin(t*n),  np.cos(n*t)]]
    return np.matmul(cw, x0)


def drag_cw(x0, times):
    r_e = 6378136.3
    mu = 3.986004415E14
    reference_orbit = OrbitalElements.OrbitalElements(r_e + 300000, 0.0, 28.5 * np.pi / 180, 0.0, 0.0, 0.0, mu)
    mu = 3.986004415E14
    alpha = .2

    xd = []
    yd = []
    zd = []

    for i in range(len(times)):
        state = st_drag_carter_humi(alpha, reference_orbit, mu, x0, times[i])
        xd.append(state[0])
        yd.append(state[1])
        zd.append(state[2])

    return [yd, xd, zd]
