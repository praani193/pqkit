import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY
from core.observable import Observable
from utils.encoding import angle_encoding


class QuantumLinearRegressor:

    def __init__(self, lr=0.1, epochs=100):
        self.theta = np.random.randn()
        self.lr = lr
        self.epochs = epochs

    def forward(self, x):

        qc = QuantumCircuit(1)

        angle_encoding(qc, x)

        qc.add_gate(RY(self.theta, 0))

        state = qc.run()

        return Observable.expectation_z(state, 0)

    def train(self, X, y):

        for epoch in range(self.epochs):

            grad = 0
            loss = 0

            for xi, yi in zip(X, y):

                pred = self.forward(xi)

                loss += (pred - yi)**2

                shift = np.pi/2

                self.theta += shift
                p1 = self.forward(xi)

                self.theta -= 2*shift
                p2 = self.forward(xi)

                self.theta += shift

                grad += (p1 - p2)/2 * 2*(pred - yi)

            self.theta -= self.lr * grad/len(X)

            if epoch % 10 == 0:
                print("Epoch", epoch, "Loss", loss/len(X))

    def predict(self, X):
        return np.array([self.forward(x) for x in X])