from main.circuit import Circuit
from main.gates import H, RZ

qc = Circuit(int(input("Enter n:")))

qc.apply(H(0))
qc.apply(RZ(1,0))

print("Statevector:", qc.get_statevector())
print("Probabilities:", qc.measure_probabilities())
print("Counts:", qc.run(shots=1000))
