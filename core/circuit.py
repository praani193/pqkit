from core.state import StateVector, BatchStateVector


class QuantumCircuit:

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    def add_gate(self, gate):
        self.gates.append(gate)


    def run(self, initial_state=None):
        """
        Run circuit on single quantum state
        """

        if initial_state is None:
            state = StateVector(self.n_qubits)
        else:
            state = initial_state

        for gate in self.gates:
            gate.apply(state)

        return state

    def run_shots(self, shots=1024):
        state = self.run()
        return state.sample(shots)

    def run_batch(self, batch_size, initial_state=None):
        """
        Run circuit on multiple states simultaneously.
        Used in Quantum ML training.
        """

        if initial_state is None:
            state = BatchStateVector(self.n_qubits, batch_size)
        else:
            state = initial_state

        for gate in self.gates:
            gate.apply(state)

        return state

    def clear(self):
        self.gates = []