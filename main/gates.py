import numpy as np


# =========================
# Base Gate Class
# =========================

class Gate:
    def __init__(self, name="GenericGate"):
        self.name = name

    def expand(self, n_qubits):
        raise NotImplementedError("Expand method must be implemented.")

    def __repr__(self):
        return f"{self.name} Gate"


# =========================
# Single Qubit Gate
# =========================

class SingleQubitGate(Gate):
    def __init__(self, matrix, target, name="SingleQubitGate"):
        super().__init__(name)
        self.matrix = matrix
        self.target = target

    def apply(self, statevector):
        state = statevector.state
        n = statevector.n_qubits
        dim = statevector.dim

        for i in range(dim):
            if ((i >> self.target) & 1) == 0:
                j = i | (1 << self.target)
                a = state[i]
                b = state[j]
                state[i] = self.matrix[0, 0] * a + self.matrix[0, 1] * b
                state[j] = self.matrix[1, 0] * a + self.matrix[1, 1] * b


# =========================
# Multi Qubit Gate (CNOT, CZ, SWAP)
# =========================

class MultiQubitGate(Gate):
    def __init__(self, name, qubits):
        super().__init__(name)
        self.qubits = qubits

    def apply(self, statevector):
        if self.name == "CNOT":
            self._apply_cnot(statevector)
        elif self.name == "CZ":
            self._apply_cz(statevector)
        elif self.name == "SWAP":
            self._apply_swap(statevector)
        else:
            raise ValueError("Unknown multi-qubit gate")

    def _apply_cnot(self, statevector):
        state = statevector.state
        n = statevector.n_qubits
        dim = statevector.dim
        control, target = self.qubits

        for i in range(dim):
            if ((i >> control) & 1) == 1:
                j = i ^ (1 << target)

                if i < j:
                    state[i], state[j] = state[j], state[i]

    def _apply_cz(self, statevector):
        state = statevector.state
        dim = statevector.dim
        control, target = self.qubits

        for i in range(dim):
            if ((i >> control) & 1) == 1 and ((i >> target) & 1) == 1:
                state[i] *= -1

    def _apply_swap(self, statevector):
        state = statevector.state
        dim = statevector.dim
        q1, q2 = self.qubits

        for i in range(dim):
            bit1 = (i >> q1) & 1
            bit2 = (i >> q2) & 1

            if bit1 != bit2:
                j = i ^ ((1 << q1) | (1 << q2))
                if i < j:
                    state[i], state[j] = state[j], state[i]

# =========================
# Single Qubit Gate APIs
# =========================

def X(target):
    matrix = np.array([[0, 1],
                       [1, 0]], dtype=complex)
    return SingleQubitGate(matrix, target, name="X")


def Y(target):
    matrix = np.array([[0, -1j],
                       [1j, 0]], dtype=complex)
    return SingleQubitGate(matrix, target, name="Y")


def Z(target):
    matrix = np.array([[1, 0],
                       [0, -1]], dtype=complex)
    return SingleQubitGate(matrix, target, name="Z")


def H(target):
    matrix = (1 / np.sqrt(2)) * np.array([[1, 1],
                                          [1, -1]], dtype=complex)
    return SingleQubitGate(matrix, target, name="H")


def RX(theta, target):
    matrix = np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RX")


def RY(theta, target):
    matrix = np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RY")


def RZ(theta, target):
    matrix = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RZ")


# =========================
# Multi Qubit Gate APIs
# =========================

def CNOT(control, target):
    return MultiQubitGate("CNOT", [control, target])


def CZ(control, target):
    return MultiQubitGate("CZ", [control, target])


def SWAP(q1, q2):
    return MultiQubitGate("SWAP", [q1, q2])
