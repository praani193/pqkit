from models.qgan import QGAN

model = QGAN(
    lr=0.05,
    epochs=1000000
)

model.train()