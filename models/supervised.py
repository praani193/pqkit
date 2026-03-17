import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY, RX, CNOT
from .utils import expectation_z


class QuantumClassifier:
    def __init__(self, n_qubits, n_params):
        self.n_qubits = n_qubits
        self.params = np.random.randn(n_params)

    def forward(self, x, params=None):
        if params is None:
            params = self.params

        qc = QuantumCircuit(self.n_qubits)

        qc.add_gate(RY(x, 0))
        qc.add_gate(RX(params[0], 0))
        qc.add_gate(RY(params[1], 1))
        qc.add_gate(CNOT(0,1))

        sv = qc.run()

        return expectation_z(sv.state, 0)