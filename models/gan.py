import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY, CNOT


class QuantumGenerator:
    def __init__(self):
        self.params = np.random.randn(1)

    def generate(self):
        qc = QuantumCircuit(2)

        qc.add_gate(RY(self.params[0],0))
        qc.add_gate(CNOT(0,1))

        sv = qc.run()

        return sv.sample(1)