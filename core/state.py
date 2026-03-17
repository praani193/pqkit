import numpy as np


class StateVector:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure_all(self):
        probs = self.probabilities()
        idx = np.random.choice(self.dim, p=probs)

        self.state = np.zeros(self.dim, dtype=complex)
        self.state[idx] = 1.0

        return format(idx, f'0{self.n_qubits}b')

    def sample(self, shots=1024):
        probs = self.probabilities()
        outcomes = np.random.choice(self.dim, shots, p=probs)

        counts = {}
        for o in outcomes:
            b = format(o, f'0{self.n_qubits}b')
            counts[b] = counts.get(b, 0) + 1
        return counts


class BatchStateVector:
    def __init__(self, n_qubits, batch_size):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.batch_size = batch_size

        # Shape = (batch , 2^n)
        self.state = np.zeros((batch_size, self.dim), dtype=complex)

        # Initialize all states to |000>
        self.state[:, 0] = 1.0