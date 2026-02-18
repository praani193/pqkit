class Gate:
    def __init__(self, matrix, name="GenericGate"):
        self.matrix = matrix
        self.name = name

    def __repr__(self):
        return f"{self.name} Gate"
