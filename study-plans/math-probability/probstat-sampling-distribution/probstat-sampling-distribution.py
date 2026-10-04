from math import sqrt
from scipy.stats import norm

def sampling_distribution(mu: float, sigma: float, n: int, threshold: float) -> dict:
    """
    Returns the sampling mean, sampling standard deviation, and lower-tail probability.
    """
    sampling_mean = mu
    sampling_std = sigma/sqrt(n)
    prob = norm.cdf(threshold, loc = sampling_mean, scale = sampling_std)
    return {"prob_below_threshold": prob, "sampling_mean": sampling_mean, "sampling_std": sampling_std}
