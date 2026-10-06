import numpy as np

def bootstrap_ci(data: list, n_bootstraps: int, confidence: float, seed: int) -> list:
    """
    Returns the bootstrap mean and percentile confidence interval.
    """
    result = []
    bootstrap_means = []
    data = np.array(data, dtype = np.float64)
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size = len(data), replace = True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_mean = np.mean(bootstrap_means)
    lower_percentile = ((1-confidence)/2)*100
    higher_percentile = ((1+confidence)/2)*100
    lower = np.percentile(bootstrap_means, lower_percentile)
    higher = np.percentile(bootstrap_means, higher_percentile)
    result.append(bootstrap_mean)
    result.append(lower)
    result.append(higher)
    return result
    
