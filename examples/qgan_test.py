import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from models.qgan import QGAN
from datasets.gaussian import sample_real_data


# Initialize model
model = QGAN(
    lr=0.05,
    epochs=10000,
    batch_size=64
)

# Train
model.train(sample_real_data)


# ===== Compare distributions =====
real = sample_real_data(1000)
fake = model.G.sample(1000)

plt.hist(real, bins=50, alpha=0.5, label="Real")
plt.hist(fake, bins=50, alpha=0.5, label="Fake")

plt.legend()
plt.title("QGAN Learning Distribution")
plt.show()