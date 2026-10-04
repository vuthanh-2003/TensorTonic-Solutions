import numpy as np
from scipy.stats import norm

def clt_confidence_interval(data: list, confidence: float) -> list:
    """
    Returns the sample mean, standard error, and confidence-interval endpoints.
    """
    results = []
    x = np.array(data, dtype = np.float64)
    z = norm.ppf(0.5 + confidence/2.0)
    mean = np.mean(x)
    std = np.std(x,ddof = 1)
    se = std/np.sqrt(len(x))
    ci_lower = mean - z*se
    ci_higher = mean + z*se
    results.append(mean)
    results.append(se)
    results.append(ci_lower)
    results.append(ci_higher)
    return results
    