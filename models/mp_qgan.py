import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY, RZ, CNOT


# ================= GENERATOR =================
class QuantumGenerator:

    def __init__(self, mode="simple", n_qubits=2, n_layers=2):

        self.mode = mode
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if mode == "simple":
            self.theta = np.random.randn(3)

        elif mode == "multi":
            self.theta = np.random.randn(n_qubits)

        elif mode == "variational":
            self.theta = np.random.randn(n_layers, n_qubits)

    def sample(self, batch_size=1):

        samples = []

        for _ in range(batch_size):

            if self.mode == "simple":
                qc = self._simple_circuit()

            elif self.mode == "multi":
                qc = self._multi_circuit()

            elif self.mode == "variational":
                noise = np.random.normal(0, 0.1, size=self.theta.shape)
                original_theta = self.theta.copy()

                self.theta = self.theta + noise
                qc = self._variational_circuit()
                self.theta = original_theta

            state = qc.run()
            samples.append(self._measure_prob(state))

        return np.array(samples)

    def _simple_circuit(self):
        qc = QuantumCircuit(1)
        qc.add_gate(RY(self.theta[0], 0))
        qc.add_gate(RZ(self.theta[1], 0))
        qc.add_gate(RY(self.theta[2], 0))
        return qc

    def _multi_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        for q in range(self.n_qubits):
            qc.add_gate(RY(self.theta[q], q))

        qc.add_gate(CNOT(0, 1))
        return qc

    def _variational_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.add_gate(RY(self.theta[layer][q], q))

            for q in range(self.n_qubits - 1):
                qc.add_gate(CNOT(q, q + 1))

        return qc

    def _measure_prob(self, state):
        prob = 0
        for i, amp in enumerate(state.state):
            if i % 2 == 1:
                prob += abs(amp)**2
        return prob


# ================= DISCRIMINATOR =================
class ClassicalDiscriminator:

    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x):
        z = self.w * x + self.b
        return 1 / (1 + np.exp(-z))


# ================= MULTI QGAN =================
class MultiQGAN:

    def __init__(self, lr_D=0.01, lr_G=0.02, epochs=200, batch_size=64):

        self.G1 = QuantumGenerator(mode="variational", n_layers=4)
        self.G2 = QuantumGenerator(mode="variational", n_layers=4)

        self.D1 = ClassicalDiscriminator()
        self.D2 = ClassicalDiscriminator()
        self.D_main = ClassicalDiscriminator()

        self.lr_D = lr_D
        self.lr_G = lr_G
        self.epochs = epochs
        self.batch_size = batch_size

    def split_real(self, real):
        low = real[real <= 0.5]
        high = real[real > 0.5]

        if len(low) == 0:
            low = real
        if len(high) == 0:
            high = real

        return low, high

    def save_plot(self, real, fake, epoch):

        import matplotlib.pyplot as plt
        import os

        os.makedirs("plots", exist_ok=True)

        plt.figure()

        plt.hist(real, bins=50, alpha=0.5, label="Real", density=True)
        plt.hist(fake, bins=50, alpha=0.6, label="Fake", density=True)

        plt.legend()
        plt.title(f"Epoch {epoch}")
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.savefig(f"plots/epoch_{epoch:04d}.png")
        plt.close()
    def train(self, real_sampler):

        for epoch in range(self.epochs):

            # ===== SAMPLE REAL =====
            real = real_sampler(self.batch_size)
            real_low, real_high = self.split_real(real)

            # ===== GENERATE FAKE =====
            fake1 = self.G1.sample(len(real_low))
            fake2 = self.G2.sample(len(real_high))

            # ===== LOCAL DISCRIMINATORS =====
            d1_real = self.D1.forward(real_low)
            d1_fake = self.D1.forward(fake1)

            d2_real = self.D2.forward(real_high)
            d2_fake = self.D2.forward(fake2)

            real_label = 0.9

            # Update D1
            self.D1.w -= self.lr_D * np.mean((d1_real - real_label)*real_low + d1_fake*fake1)
            self.D1.b -= self.lr_D * np.mean((d1_real - real_label) + d1_fake)

            # Update D2
            self.D2.w -= self.lr_D * np.mean((d2_real - real_label)*real_high + d2_fake*fake2)
            self.D2.b -= self.lr_D * np.mean((d2_real - real_label) + d2_fake)

            # ===== GENERATOR (LOCAL) =====
            for _ in range(2):
                self._update_generator(self.G1, self.D1, fake1)
                self._update_generator(self.G2, self.D2, fake2)

            # ===== GLOBAL =====
            fake_all = np.concatenate([fake1, fake2])

            d_real = self.D_main.forward(real)
            d_fake = self.D_main.forward(fake_all)

            self.D_main.w -= self.lr_D * np.mean((d_real - real_label)*real + d_fake*fake_all)
            self.D_main.b -= self.lr_D * np.mean((d_real - real_label) + d_fake)

            # GLOBAL GENERATOR
            for _ in range(2):
                self._update_generator(self.G1, self.D_main, fake1)
                self._update_generator(self.G2, self.D_main, fake2)

            # ===== LOG =====
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}")
                print("---------- LOCAL ----------")
                print(f"D1(real): {np.mean(d1_real):.3f} | D1(fake): {np.mean(d1_fake):.3f}")
                print(f"D2(real): {np.mean(d2_real):.3f} | D2(fake): {np.mean(d2_fake):.3f}")

                print("---------- GLOBAL ----------")
                print(f"D_main(real): {np.mean(d_real):.3f} | D_main(fake): {np.mean(d_fake):.3f}")

                loss_d = -np.mean(np.log(d_real + 1e-8)) - np.mean(np.log(1 - d_fake + 1e-8))
                loss_g = -np.mean(np.log(d_fake + 1e-8))

                print("---------- LOSSES ----------")
                print(f"D Loss: {loss_d:.4f}")
                print(f"G Loss: {loss_g:.4f}")
                real_plot = real_sampler(500)
                fake_plot = np.concatenate([
                    self.G1.sample(250),
                    self.G2.sample(250)
                ])

                self.save_plot(real_plot, fake_plot, epoch)

        # ===== FINAL PLOT (ONLY ONE) =====
        real_samples = real_sampler(1000)
        fake_samples = np.concatenate([
            self.G1.sample(500),
            self.G2.sample(500)
        ])
        # encourage diversity
        if np.std(fake1) < 0.05:
            self.G1.theta += np.random.randn(*self.G1.theta.shape) * 0.05

        if np.std(fake2) < 0.05:
            self.G2.theta += np.random.randn(*self.G2.theta.shape) * 0.05
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))

        plt.hist(real_samples, bins=50, alpha=0.5, label="Real", density=True)
        plt.hist(fake_samples, bins=50, alpha=0.5, label="Fake", density=True)

        plt.legend()
        plt.title("Final Real vs Fake Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.show()

    # ===== GENERATOR UPDATE =====
    def _update_generator(self, G, D, fake_batch):

        shift = np.pi / 2

        G.theta += shift
        fake_plus = G.sample(len(fake_batch))

        G.theta -= 2 * shift
        fake_minus = G.sample(len(fake_batch))

        G.theta += shift

        d_fake = D.forward(fake_batch)

        grad = np.mean((fake_plus - fake_minus) / 2 * (-1 / (1 - d_fake + 1e-8)))

        G.theta -= self.lr_G * grad