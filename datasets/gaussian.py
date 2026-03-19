import numpy as np

def sample_real_data(batch_size=1):
    return np.random.normal(loc=0.7, scale=0.1, size=batch_size)
