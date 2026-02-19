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

    def expand(self, n_qubits):
        I = np.eye(2, dtype=complex)
        operators = []

        for i in range(n_qubits):
            if i == self.target:
                operators.append(self.matrix)
            else:
                operators.append(I)

        U = operators[0]
        for op in operators[1:]:
            U = np.kron(U, op)

        return U


# =========================
# Multi Qubit Gate (CNOT, CZ, SWAP)
# =========================

class MultiQubitGate(Gate):
    def __init__(self, name, qubits):
        super().__init__(name)
        self.qubits = qubits  # list of involved qubits

    def expand(self, n_qubits):
        if self.name == "CNOT":
            return self._expand_cnot(n_qubits)
        elif self.name == "CZ":
            return self._expand_cz(n_qubits)
        elif self.name == "SWAP":
            return self._expand_swap(n_qubits)
        else:
            raise ValueError(f"Unknown multi-qubit gate: {self.name}")

    # -------------------------
    # CNOT
    # -------------------------

    def _expand_cnot(self, n):
        control, target = self.qubits
        dim = 2 ** n
        matrix = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            binary = list(format(i, f'0{n}b'))

            if binary[control] == '1':
                binary[target] = '0' if binary[target] == '1' else '1'

            j = int("".join(binary), 2)
            matrix[j, i] = 1

        return matrix

    # -------------------------
    # CZ
    # -------------------------

    def _expand_cz(self, n):
        control, target = self.qubits
        dim = 2 ** n
        matrix = np.eye(dim, dtype=complex)

        for i in range(dim):
            binary = list(format(i, f'0{n}b'))

            if binary[control] == '1' and binary[target] == '1':
                matrix[i, i] = -1

        return matrix

    # -------------------------
    # SWAP
    # -------------------------

    def _expand_swap(self, n):
        q1, q2 = self.qubits
        dim = 2 ** n
        matrix = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            binary = list(format(i, f'0{n}b'))

            binary[q1], binary[q2] = binary[q2], binary[q1]

            j = int("".join(binary), 2)
            matrix[j, i] = 1

        return matrix


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
