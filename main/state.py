import numpy as np
import random


class StateVector:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)

        # Initialize |000...0>
        self.state[0] = 1.0

    def apply_gate(self, gate):
        gate.apply(self)

    def probabilities(self):
        return np.abs(self.state) ** 2

    # ---------------------------
    # Measure single qubit
    # ---------------------------
    def measure_qubit(self, qubit):
        prob_0 = 0
        prob_1 = 0

        for i in range(self.dim):
            if ((i >> qubit) & 1) == 0:
                prob_0 += abs(self.state[i]) ** 2
            else:
                prob_1 += abs(self.state[i]) ** 2

        result = np.random.choice([0, 1], p=[prob_0, prob_1])

        # Collapse
        for i in range(self.dim):
            if ((i >> qubit) & 1) != result:
                self.state[i] = 0

        # Renormalize
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

        return result

    # ---------------------------
    # Measure all qubits
    # ---------------------------
    def measure_all(self):
        probs = self.probabilities()
        index = np.random.choice(self.dim, p=probs)

        bitstring = format(index, f'0{self.n_qubits}b')

        # Collapse fully
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[index] = 1.0

        return bitstring

    # ---------------------------
    # Shot-based sampling
    # ---------------------------
    def sample(self, shots=1024):
        probs = self.probabilities()
        outcomes = np.random.choice(self.dim, size=shots, p=probs)

        counts = {}
        for idx in outcomes:
            bitstring = format(idx, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def __repr__(self):
        return str(self.state)