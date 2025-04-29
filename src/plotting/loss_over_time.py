import numpy as np
from scipy.ndimage import uniform_filter1d

def compute_mean_and_std_losses(losses: np.array, window_size: int):
    """Smoothes an array and computes mean and variance of an array in log-scale."""
    
    smoothed_losses = uniform_filter1d(losses, size=window_size, axis=1, mode='reflect')
    mean_losses_smoothed = smoothed_losses.mean(axis=0)
    log_std_losses_smoothed = np.exp(np.std(np.log(smoothed_losses + 1e-8), axis=0))
    return mean_losses_smoothed, log_std_losses_smoothed