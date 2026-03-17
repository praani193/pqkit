import numpy as np


def expectation_z(state, qubit):
    val = 0
    dim = len(state)

    for i in range(dim):
        bit = (i >> qubit) & 1
        sign = 1 if bit == 0 else -1
        val += sign * abs(state[i])**2

    return val


def parameter_shift(model, x, params, i):
    shift = np.pi/2

    p_plus = params.copy()
    p_minus = params.copy()

    p_plus[i] += shift
    p_minus[i] -= shift

    return (model.forward(x, p_plus) -
            model.forward(x, p_minus)) / 2