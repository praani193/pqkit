import numpy as np

class StateVector:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

    def apply_gate(self, gate):
        gate.apply(self)

    def measure_probabilities(self):
        return np.abs(self.state) ** 2

    def __repr__(self):
        return str(self.state)
