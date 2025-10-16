import numpy as np
from pathlib import Path

def read_noise(noise_file: Path, n_samples: int):
    with open(noise_file) as f:
        file_contents = f.read()
    const_noise = np.fromstring(file_contents, sep=' ')
    if const_noise.size == 1:
        return True, const_noise[0]
    if const_noise.shape[0] != n_samples:
        raise ValueError("Length of const_noise does not match number of samples in X_val_filtered.")
    return False, const_noise
