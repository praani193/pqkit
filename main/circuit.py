import numpy as np
from .state import QuantumState


class Circuit:
    def __init__(self, n_qubits):
        if n_qubits < 1:
            raise ValueError("Number of qubits must be >= 1")

        self.n = n_qubits
        self.state = QuantumState(n_qubits)

    # ======================================================
    # Apply a Gate
    # ======================================================

    def apply(self, gate):
        """
        Apply a Gate object to the circuit.
        """
        U = gate.expand(self.n)
        self.state.apply_unitary(U)

    # ======================================================
    # Measurement
    # ======================================================

    def measure_probabilities(self):
        """
        Return probability distribution of current state.
        """
        return self.state.probabilities()

    # ======================================================
    # Run with Shots
    # ======================================================

    def run(self, shots=1024):
        """
        Sample measurement results.
        """
        probs = self.measure_probabilities()

        if not np.isclose(np.sum(probs), 1):
            raise ValueError("State is not normalized.")

        outcomes = np.arange(len(probs))

        samples = np.random.choice(outcomes, size=shots, p=probs)

        counts = {}
        for s in samples:
            bitstring = format(s, f'0{self.n}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    # ======================================================
    # Reset Circuit
    # ======================================================

    def reset(self):
        """        Reset to |00...0>
        """
        self.state = QuantumState(self.n)

    # ======================================================
    # Debug Info
    # ======================================================

    def get_statevector(self):
        return self.state.state

    def __repr__(self):
        return f"Circuit(n_qubits={self.n})"
