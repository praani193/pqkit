import numpy as np
from core.circuit import QuantumCircuit
from core.gates import RY


# ================= GENERATOR =================
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

    def __init__(self, lr=0.05, epochs=300, batch_size=64):

        self.G1 = QuantumGenerator()
        self.G2 = QuantumGenerator()

        self.D1 = ClassicalDiscriminator()
        self.D2 = ClassicalDiscriminator()

        self.D_main = ClassicalDiscriminator()

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    # ---------- REAL DATA SPLIT ----------
    def split_real(self, real_batch):
        low = real_batch[real_batch <= 0.5]
        high = real_batch[real_batch > 0.5]

        # avoid empty batches
        if len(low) == 0:
            low = real_batch
        if len(high) == 0:
            high = real_batch

        return low, high

    # ---------- TRAIN ----------
    def train(self, real_sampler):

        for epoch in range(self.epochs):

            # ===== SAMPLE REAL =====
            real = real_sampler(self.batch_size)
            real_low, real_high = self.split_real(real)

            # ===== GENERATE FAKE =====
            fake1 = self.G1.sample(len(real_low))
            fake2 = self.G2.sample(len(real_high))

            # ===== TRAIN LOCAL DISCRIMINATORS =====
            d1_real = self.D1.forward(real_low)
            d1_fake = self.D1.forward(fake1)

            d2_real = self.D2.forward(real_high)
            d2_fake = self.D2.forward(fake2)

            # gradients
            self.D1.w -= self.lr * np.mean((d1_real - 1)*real_low + d1_fake*fake1)
            self.D1.b -= self.lr * np.mean((d1_real - 1) + d1_fake)

            self.D2.w -= self.lr * np.mean((d2_real - 1)*real_high + d2_fake*fake2)
            self.D2.b -= self.lr * np.mean((d2_real - 1) + d2_fake)

            # ===== TRAIN GENERATORS (LOCAL) =====
            self._update_generator(self.G1, self.D1, fake1)
            self._update_generator(self.G2, self.D2, fake2)

            # ===== GLOBAL TRAINING =====
            fake_all = np.concatenate([fake1, fake2])

            d_real = self.D_main.forward(real)
            d_fake = self.D_main.forward(fake_all)

            # update master
            # Separate gradients
            grad_real = (d_real - 1) * real
            grad_fake = d_fake * fake_all

            self.D_main.w -= self.lr * (np.mean(grad_real) + np.mean(grad_fake))
            self.D_main.b -= self.lr * (np.mean(d_real - 1) + np.mean(d_fake))

            # update generators globally
            self._update_generator(self.G1, self.D_main, fake1)
            self._update_generator(self.G2, self.D_main, fake2)

            if epoch % 10 == 0:
                # Local metrics
                d1_real_mean = np.mean(d1_real)
                d1_fake_mean = np.mean(d1_fake)

                d2_real_mean = np.mean(d2_real)
                d2_fake_mean = np.mean(d2_fake)

                # Global metrics
                d_real_mean = np.mean(d_real)
                d_fake_mean = np.mean(d_fake)

                # Losses
                loss_real = -np.mean(np.log(d_real + 1e-8))
                loss_fake = -np.mean(np.log(1 - d_fake + 1e-8))

                loss_d_main = loss_real + loss_fake
                loss_g = -np.mean(np.log(d_fake + 1e-8))

                print(f"\nEpoch {epoch}")
                print("---------- LOCAL ----------")
                print(f"D1(real): {d1_real_mean:.3f} | D1(fake): {d1_fake_mean:.3f}")
                print(f"D2(real): {d2_real_mean:.3f} | D2(fake): {d2_fake_mean:.3f}")

                print("---------- GLOBAL ----------")
                print(f"D_main(real): {d_real_mean:.3f} | D_main(fake): {d_fake_mean:.3f}")

                print("---------- LOSSES ----------")
                print(f"D Loss: {loss_d_main:.4f}")
                print(f"G Loss: {loss_g:.4f}")

    # ---------- PARAMETER SHIFT UPDATE ----------
    def _update_generator(self, G, D, fake_batch):

        shift = np.pi / 2

        G.theta += shift
        fake1 = G.sample(len(fake_batch))

        G.theta -= 2 * shift
        fake2 = G.sample(len(fake_batch))

        G.theta += shift

        d_fake = D.forward(fake_batch)

        grad = np.mean((fake1 - fake2)/2 * (-1/(1 - d_fake + 1e-8)))

        G.theta -= self.lr * grad