import numpy as np


class Observable:

    @staticmethod
    def expectation_z(statevector, qubit=0):
        state = statevector.state
        dim = statevector.dim

        exp = 0

        for i in range(dim):
            prob = abs(state[i])**2
            bit = (i >> qubit) & 1

            if bit == 0:
                exp += prob
            else:
                exp -= prob

        return exp

    @staticmethod
    def expectation_x(statevector, qubit=0):
        state = statevector.state
        dim = statevector.dim

        exp = 0

        for i in range(dim):
            j = i ^ (1 << qubit)
            exp += np.conj(state[i]) * state[j]

        return np.real(exp)

    @staticmethod
    def expectation_y(statevector, qubit=0):
        state = statevector.state
        dim = statevector.dim

        exp = 0

        for i in range(dim):
            j = i ^ (1 << qubit)

            bit = (i >> qubit) & 1

            if bit == 0:
                exp += -1j * np.conj(state[i]) * state[j]
            else:
                exp += 1j * np.conj(state[i]) * state[j]

        return np.real(exp)