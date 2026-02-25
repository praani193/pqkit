from main.gates import H, CNOT
from main.circuit import QuantumCircuit

qc = QuantumCircuit(2)
qc.add_gate(H(0))
qc.add_gate(CNOT(0, 1))

# Single run
state = qc.run()
print("Statevector:", state.state)

# Single measurement
print("Measured:", state.measure_all())

# Shot-based execution
counts = qc.run_shots(1000)
print("Counts:", counts)