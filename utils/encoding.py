from core.gates import RX


def angle_encoding(circuit, x):
    circuit.add_gate(RX(x, 0))