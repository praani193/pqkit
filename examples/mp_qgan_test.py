import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from models.mp_qgan import MultiQGAN
from datasets.gaussian_new import sample_real_data


model = MultiQGAN(
        epochs=501,
        batch_size=64
    )

model.train(sample_real_data)

print(model.G1.theta, model.G2.theta)
# ===== Compare distributions =====


""""
real = sample_real_data(1000)
fake1 = model.G1.sample(500)

import matplotlib.pyplot as plt

plt.hist(real, bins=50, alpha=0.5, label="Real")
plt.hist(list(fake1), bins=50, alpha=0.5, label="Fake")

plt.legend()
plt.show()
"""