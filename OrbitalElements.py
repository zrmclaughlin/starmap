import numpy as np
import Transformation
import PropagateOrbit


class OrbitalElements:

    def __init__(self, a, e, i, w, _o, _nu, mu):

        self._a = a
        self._e = e
        self._i = i
        self._w = w
        self._o = _o
        self._nu = _nu
        self.mu = mu

    def set_nu(self, _nu):
        if _nu < 0:
            self._nu = (_nu % (-2*np.pi)) + 2*np.pi
        else:
            self._nu = _nu % (2*np.pi)

    def get_nu(self):
        return self._nu

    def get_a(self):
        return self._a

    def set_a(self, a):
        self._a = a

    def get_e(self):
        return self._e

    def set_e(self, e):
        self._e = e

    def get_i(self):
        return self._i

    def set_i(self, i):
        self._i = i % (2 * np.pi)

    def get_w(self):
        return self._w

    def set_w(self, w):
        self._w = w % (2 * np.pi)

    def get_o(self):
        return self._o

    def set_o(self, o):
        self._o = o % (2*np.pi)

    def get_cartesian(self):

        r_scalar = self._a * (1 - self._e ** 2) / (1 + self._e * np.cos(self._nu))
        rx = r_scalar * np.cos(self._nu)
        ry = r_scalar * np.sin(self._nu)
        rz = 0

        v_scalar = np.sqrt(self.mu / (self._a * (1 - self._e ** 2)))
        vx = v_scalar * (-np.sin(self._nu))
        vy = v_scalar * (self._e + np.cos(self._nu))
        vz = 0

        if (self._e == 0) & (self._i == 0):
            r_313_transform = np.identity(3)
        elif (self._e == 0):
            r_313_transform = np.matmul(Transformation.t_3(-self._o), Transformation.t_1(-self._i))
        elif (self._i == 0):
            r_313_transform = Transformation.t_3(-self._w)
        else:
            r_313_transform = np.matmul(Transformation.t_3(-self._o),
                              np.matmul(Transformation.t_1(-self._i),
                                        Transformation.t_3(-self._w)))

        return [np.matmul(r_313_transform, np.asarray([rx, ry, rz])),
                np.matmul(r_313_transform, np.asarray([vx, vy, vz]))]


def from_cartesian(r, v, mu):
    # find the specific energy of the orbiting body
    energy = (np.linalg.norm(v) * np.linalg.norm(v) / 2) - mu / np.linalg.norm(r)

    # Set the semi-major axis
    a = - mu / (2 * energy)

    # find the angular momentum and the line of nodes (and its unit vector)
    h = np.cross(np.asarray(r), np.asarray(v))
    n = np.cross(np.asarray([0, 0, 1]), h)
    n_hat = (1 / np.linalg.norm(n)) * np.asarray(n)

    # find the eccentricity vector
    v_cross_h = np.cross(v, h)
    e_vec = np.subtract((1 / mu) * np.asarray(v_cross_h), (1 / np.linalg.norm(r) * np.asarray(r)))

    # Set the eccentricity
    e = np.linalg.norm(e_vec)

    # Set the inclination
    i = np.arccos(np.dot(np.asarray([0, 0, 1]), h) / np.linalg.norm(h))

    # Set the right ascension of the ascending node
    o = np.arccos(n_hat[0])  # RAAN
    if n[1] < 0:  # if the y element of the line of nodes is below 0, subtract from 2pi
        o = 2 * np.pi - o

    # Set the argument of periapse
    w = np.arccos(np.dot(np.asarray(n), e_vec) / (np.linalg.norm(n) * e))  # Argument of Periapse
    if e_vec[2] < 0:  # If the z element uof the eccentricity is below 0, subtract from 2pi
        w = 2 * np.pi - w

    # Set _nu
    nu = np.arccos((a * (1 - e ** 2) - np.linalg.norm(r)) / (e * np.linalg.norm(r)))
    if np.dot(r, v) < 0:  # if the dot product of r and v is below zero, subtract from 2*pi
        nu = 2 * np.pi - nu

    return OrbitalElements(a, e, i, w, o, nu, mu)
