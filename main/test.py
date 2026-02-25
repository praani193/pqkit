from main.gates import H, CNOT
from main.circuit import QuantumCircuit

qc = QuantumCircuit(5)
qc.add_gate(H(0))
qc.add_gate(CNOT(0, 1))

final_state = qc.run()

print(final_state.state)
print(final_state.measure_probabilities())