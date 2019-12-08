import OrbitalElements


class RelativeState:

    def __init__(self):

        self.reference_orbit = OrbitalElements.OrbitalElements(0.0, 0.0, 0.0, 0.0, 0.0)


def create_state_from_eci_xyz(x, y, z, x_dot, y_dot, z_dot,
                              x_chaser_rtn, y_chaser_rtn, z_chaser_rtn,
                              x_dot_chaser_rtn, y_dot_chaser_rtn, z_dot_chaser_rtn):
    return


def create_state_from_orbit(semi_major_axis, raan, inclination, true_anomaly, eccentricity,
                            x_chaser_rtn, y_chaser_rtn, z_chaser_rtn,
                            x_dot_chaser_rtn, y_dot_chaser_rtn, z_dot_chaser_rtn):
    return