import numpy as np
def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=arr)

def compute_mean_and_std_losses(losses: np.array, window_size: int):
    """Smoothes an array and computes mean and variance of an array in log-scale.
    Args:
    losses: np.array of shape (n_runs, n_optimization_steps)
    window_size: int
    """

    
    mean_losses_smoothed = moving_average(losses, window_size).mean(axis=0)
    log_std_losses_smoothed = np.exp(np.std(np.log(moving_average(losses, window_size) + 1e-8), axis=0))
    return mean_losses_smoothed, log_std_losses_smoothed