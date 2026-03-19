import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from models.mp_qgan import MultiQGAN
from datasets.gaussian import sample_real_data


model = MultiQGAN(
    lr=0.05,
    epochs=3000,
    batch_size=64
)

model.train(sample_real_data)


# ===== Compare distributions =====

real = sample_real_data(1000)
fake1 = model.G1.sample(500)
fake2 = model.G2.sample(500)

import matplotlib.pyplot as plt

plt.hist(real, bins=50, alpha=0.5, label="Real")
plt.hist(list(fake1)+list(fake2), bins=50, alpha=0.5, label="Fake")

plt.legend()
plt.show()