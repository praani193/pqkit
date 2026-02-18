import numpy as np


class QuantumState:
    def __init__(self, n):
        self.n = n
        self.dim = 2**n
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
    def apply_unitary(self, U):
        self.state = U @ self.state
    def probabilities(self):
        return np.abs(self.state) ** 2

    def __repr__(self):
        return f"QuantumState(n={self.n}, state={self.state})"
