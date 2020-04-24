import numpy as np
from scipy import special


def perform_signs_test(data: np.ndarray, median_h0: float, alpha_level: float):
    """
    Performs signs tests. Tests expected median of population

    :param data: numpy array of shape (n_samples, sample_size)
    :param median_h0: expected median of population
    :param alpha_level: significance
    :return: Tuple of calculated test of shape (n_samples), pvalue of shape (n_samples) and power of shape (n_samples)
     in each sample
    """
    n_samples, sample_size = data.shape

    # In each sample count number of elements greater than H0 median
    greater_count = np.sum(data > median_h0, axis=1)
    assert list(greater_count.shape) == [n_samples]

    # Calculate p-value using binomial distribution for each sample
    pvalue_calculator = np.vectorize(lambda x: special.binom(sample_size, x) / (2 ** sample_size))
    pvalue = pvalue_calculator(greater_count)
    assert list(pvalue.shape) == [n_samples]

    # Calculate power for each sample
    power = np.abs(greater_count / sample_size - 0.5) > alpha_level
    power = power.astype(int)
    assert list(power.shape) == [n_samples]

    # Calculate the test
    test_results = pvalue > alpha_level
    assert list(test_results.shape) == [n_samples]

    return test_results, pvalue, power


def perform_arrange_test(data: np.ndarray, alpha_level: float, rnd=np.random) -> tuple:
    """
    Performs arrangement test. Tests that expected value of population is zero

    :param data: numpy 2d array of shape (n_samples, sample_size)
    :param alpha_level: significance
    :param rnd: numpy random state as an instrument for predictable and repeatable results
    :return: tuple of calculated test of shape (n_samples), pvalue of shape (n_samples) and power of shape (n_samples)
     in each sample
    """
    n_samples, sample_size = data.shape

    pvalue = np.zeros(n_samples, dtype=float)
    power = np.zeros(n_samples, dtype=float)

    # Calculate sum in each input sample
    sample_sum = np.sum(data, axis=1)
    assert list(sample_sum.shape) == [n_samples]

    # Run test sample_size times for average values
    for i in range(sample_size):
        # Creates random array only with values -1 and 1
        random_values = rnd.rand(n_samples, sample_size)
        assert np.max(random_values) <= 1.0 and np.min(random_values) >= 0.0
        arranger = (random_values > 0.5).astype(int) * 2 - 1
        assert list(arranger.shape) == list(data.shape)
        assert list(np.unique(arranger)) == [-1, 1]

        # Calculate sum in each randomly arranged sample
        arranged_data_sum = np.sum(data * arranger, axis=1)
        assert list(arranged_data_sum.shape) == [n_samples]

        # Calculate p-value for each sample
        pvalue += np.abs(arranged_data_sum) >= np.abs(sample_sum)

        # Calculate power for each sample
        power += (np.abs(arranged_data_sum - sample_sum) / sample_size) >= alpha_level

    pvalue /= sample_size
    power /= sample_size

    # Calculate the test
    test_results = pvalue > alpha_level
    assert list(test_results.shape) == [n_samples]

    return test_results, pvalue, power
