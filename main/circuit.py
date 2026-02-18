import numpy as np
from .state import QuantumState
from .gates import H, X, RX, CNOT
from .expansion import expand_single_qubit_gate

class Circuit:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = QuantumState(n_qubits)

    def h(self, target):
        U = expand_single_qubit_gate(H, target, self.n)
        self.state.apply_unitary(U)

    def x(self, target):
        U = expand_single_qubit_gate(X, target, self.n)
        self.state.apply_unitary(U)

    def rx(self, theta, target):
        U = expand_single_qubit_gate(RX(theta), target, self.n)
        self.state.apply_unitary(U)

    def cnot(self):
        if self.n != 2:
            raise ValueError("Basic version supports 2 qubits only.")
        self.state.apply_unitary(CNOT)

    def measure_probabilities(self):
        return self.state.probabilities()

    def run(self, shots=1024):
        probs = self.measure_probabilities()
        outcomes = np.arange(len(probs))

        samples = np.random.choice(outcomes, size=shots, p=probs)

        counts = {}
        for s in samples:
            bitstring = format(s, f'0{self.n}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
