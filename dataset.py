import numpy as np


def create_dataset(n_samples: int, sample_size: int, mean: float, sigma: float) -> np.ndarray:
    """
    Creates numpy 2d array with samples from normal distribution population

    :param n_samples: number of samples
    :param sample_size: number of elements in sample
    :param mean: mean of normal distribution
    :param sigma: standard deviation of normal distribution
    :return: numpy array of size (n_samples, sample_size)
    """

    return np.random.normal(mean, sigma, (n_samples, sample_size))
