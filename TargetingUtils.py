import numpy as np


def get_S_T_rv(flat_state):
    S_T_rv = np.ndarray([[flat_state[9], flat_state[10], flat_state[11]],
                         [flat_state[15], flat_state[16], flat_state[17]],
                         [flat_state[21], flat_state[22], flat_state[23]]])
    return S_T_rv  # Return the propagated state transition matrix in a readable fashion


def get_S_T_vv(flat_state):
    S_T_vv = np.ndarray([[flat_state[27], flat_state[28], flat_state[29]],
                         [flat_state[33], flat_state[34], flat_state[35]],
                         [flat_state[39], flat_state[40], flat_state[41]]])
    return S_T_vv  # Return the propagated state transition matrix in a readable fashion


def recompose(flat_state, state_length):
    flat_state = flat_state.tolist()
    for i in range(state_length):
        flat_state.pop(i)
    S_T = np.asarray(flat_state).reshape(state_length, state_length)
    return S_T  # Return the S_T as type ndarray
