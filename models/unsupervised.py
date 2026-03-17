import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY, CNOT


class QuantumAutoencoder:
    def __init__(self, n_qubits):
        self.params = np.random.randn(2)

    def encode(self, x):
        qc = QuantumCircuit(2)

        qc.add_gate(RY(x,0))
        qc.add_gate(RY(self.params[0],0))
        qc.add_gate(CNOT(0,1))

        return qc.run()