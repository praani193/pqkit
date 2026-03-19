from models.qgan import QGAN

model = QGAN(
    lr=0.05,
    epochs=300
)

model.train()