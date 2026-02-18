import numpy as np
class Gate:
    def __init__(self, matrix, name="GenericGate"):
        self.matrix = matrix
        self.name = name

    def expand(self, n_qubits):
        raise NotImplementedError("Expand method must be implemented.")

    def __repr__(self):
        return f"{self.name} Gate"

class SingleQubitGate(Gate):
    def __init__(self, matrix, target, name="SingleQubitGate"):
        super().__init__(matrix, name)
        self.target = target

    def expand(self, n_qubits):
        I = np.eye(2, dtype=complex)
        ops = []

        for i in range(n_qubits):
            if i == self.target:
                ops.append(self.matrix)
            else:
                ops.append(I)

        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)

        return U

class TwoQubitGate(Gate):
    def __init__(self, matrix, control, target, name="TwoQubitGate"):
        super().__init__(matrix, name)
        self.control = control
        self.target = target

    def expand(self, n_qubits):
        if n_qubits != 2:
            raise NotImplementedError(
                "Phase A supports two-qubit gates only in 2-qubit systems."
            )

        return self.matrix

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
    matrix = (1/np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex)
    return SingleQubitGate(matrix, target, name="H")

def RX(theta, target):
    matrix = np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RX")

def RY(theta, target):
    matrix = np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RY")

def RZ(theta, target):
    matrix = np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)
    return SingleQubitGate(matrix, target, name="RZ")


def CNOT(control=0, target=1):
    matrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]
    ], dtype=complex)
    return TwoQubitGate(matrix, control, target, name="CNOT")

def CZ(control=0, target=1):
    matrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]
    ], dtype=complex)
    return TwoQubitGate(matrix, control, target, name="CZ")
def SWAP(q1=0, q2=1):
    matrix = np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], dtype=complex)
    return TwoQubitGate(matrix, q1, q2, name="SWAP")
