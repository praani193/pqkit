import numpy as np
from main.gates import I

def expand_single_qubit_gate(gate, target, n_qubits):
    ops = []

    for i in range(n_qubits):
        if i == target:
            ops.append(gate)
        else:
            ops.append(I)

    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)

    return U
