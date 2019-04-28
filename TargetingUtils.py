import numpy as np

def get_S_T_rv(flat_state):
    S_T_rv = np.matrix([[flat_state[9], flat_state[10], flat_state[11]],
                     [flat_state[15], flat_state[16], flat_state[17]],
                     [flat_state[21], flat_state[22], flat_state[23]]])
    return S_T_rv  # Return the propagated state transition matrix in a readable fashion


def get_S_T_vv(flat_state):
    S_T_vv = np.matrix([[flat_state[27], flat_state[28], flat_state[29]],
                     [flat_state[33], flat_state[34], flat_state[35]],
                     [flat_state[39], flat_state[40], flat_state[41]]])
    return S_T_vv  # Return the propagated state transition matrix in a readable fashion


def recompose(flat_state):
    S_T = np.matrix([[flat_state[6], flat_state[7], flat_state[8], flat_state[9], flat_state[10], flat_state[11]],
                     [flat_state[12], flat_state[13], flat_state[14], flat_state[15], flat_state[16], flat_state[17]],
                     [flat_state[18], flat_state[19], flat_state[20], flat_state[21], flat_state[22], flat_state[23]],
                     [flat_state[24], flat_state[25], flat_state[26], flat_state[27], flat_state[28], flat_state[29]],
                     [flat_state[30], flat_state[31], flat_state[32], flat_state[33], flat_state[34], flat_state[35]],
                     [flat_state[36], flat_state[37], flat_state[38], flat_state[39], flat_state[40], flat_state[41]]])
    return S_T  # Return the prop