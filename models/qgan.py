import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY


# ---------------- GENERATOR ----------------
class QuantumGenerator:

    def __init__(self):
        self.theta = np.random.randn()

    def sample(self, batch_size=1):

        samples = []

        for _ in range(batch_size):
            qc = QuantumCircuit(1)
            qc.add_gate(RY(self.theta, 0))

            state = qc.run()

            prob1 = abs(state.state[1])**2
            samples.append(prob1)

        return np.array(samples)


# ---------------- DISCRIMINATOR ----------------
class ClassicalDiscriminator:

    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x):
        z = self.w * x + self.b
        return 1 / (1 + np.exp(-z))


# ---------------- QGAN ----------------
class QGAN:

    def __init__(self, lr=0.05, epochs=300, batch_size=32):
        self.G = QuantumGenerator()
        self.D = ClassicalDiscriminator()

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, real_sampler):

        for epoch in range(self.epochs):

            # ===== Sample Data =====
            real_batch = real_sampler(self.batch_size)
            fake_batch = self.G.sample(self.batch_size)

            # ===== Train Discriminator =====
            d_real = self.D.forward(real_batch)
            d_fake = self.D.forward(fake_batch)

            loss_d = -np.mean(np.log(d_real + 1e-8) +
                             np.log(1 - d_fake + 1e-8))

            grad_w = np.mean((d_real - 1)*real_batch + d_fake*fake_batch)
            grad_b = np.mean((d_real - 1) + d_fake)

            self.D.w -= self.lr * grad_w
            self.D.b -= self.lr * grad_b

            # ===== Train Generator =====
            shift = np.pi / 2

            self.G.theta += shift
            fake1 = self.G.sample(self.batch_size)

            self.G.theta -= 2 * shift
            fake2 = self.G.sample(self.batch_size)

            self.G.theta += shift

            d_fake = self.D.forward(fake_batch)

            grad_g = np.mean((fake1 - fake2)/2 * (-1/(1 - d_fake + 1e-8)))

            self.G.theta -= self.lr * grad_g

            if epoch % 20 == 0:
                print(f"Epoch {epoch} | D Loss: {loss_d:.4f}")