import numpy as np

def sample_real_data(batch_size=64):
    return np.random.normal(loc=0.6, scale=0.15, size=batch_size)
