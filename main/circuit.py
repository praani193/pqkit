from main.state import StateVector


class QuantumCircuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    def add_gate(self, gate):
        self.gates.append(gate)

    def run(self):
        state = StateVector(self.n_qubits)

        for gate in self.gates:
            gate.apply(state)

        return state

    def run_shots(self, shots=1024):
        state = self.run()
        return state.sample(shots)