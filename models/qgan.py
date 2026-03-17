import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY
from core.observable import Observable


class QuantumGenerator:

    def __init__(self):
        self.theta = np.random.randn()

    def sample(self):

        qc = QuantumCircuit(1)

        qc.add_gate(RY(self.theta, 0))

        state = qc.run()

        # Probability of |1>
        prob1 = abs(state.state[1])**2

        return prob1


class ClassicalDiscriminator:

    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x):
        z = self.w * x + self.b
        return 1 / (1 + np.exp(-z))


class QGAN:

    def __init__(self, lr=0.1, epochs=200):

        self.G = QuantumGenerator()
        self.D = ClassicalDiscriminator()

        self.lr = lr
        self.epochs = epochs

    def real_data(self):
        return np.random.normal(0.8, 0.05)

    def train(self):

        for epoch in range(self.epochs):

            # ----- Train Discriminator -----
            real = self.real_data()
            fake = self.G.sample()

            d_real = self.D.forward(real)
            d_fake = self.D.forward(fake)

            loss_d = -np.log(d_real) - np.log(1 - d_fake)

            grad_w = (d_real - 1)*real + d_fake*fake
            grad_b = (d_real - 1) + d_fake

            self.D.w -= self.lr * grad_w
            self.D.b -= self.lr * grad_b

            # ----- Train Generator -----
            shift = np.pi/2

            self.G.theta += shift
            fake1 = self.G.sample()

            self.G.theta -= 2*shift
            fake2 = self.G.sample()

            self.G.theta += shift

            grad_g = (fake1 - fake2)/2 * (-1/(1-d_fake))

            self.G.theta -= self.lr * grad_g

            if epoch % 20 == 0:
                print("Epoch", epoch, "D Loss", loss_d)