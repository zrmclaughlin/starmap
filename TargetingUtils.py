import numpy as np

def recompose(flat_state, state_length):
    flat_state = flat_state.tolist()
    for i in range(state_length):
        flat_state.pop(0)
    S_T = np.asarray(flat_state).reshape(state_length, state_length)
    return S_T  # Return the S_T as type ndarray
