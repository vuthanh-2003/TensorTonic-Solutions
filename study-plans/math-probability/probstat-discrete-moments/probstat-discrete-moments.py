import numpy as np

def discrete_moments(values: list, probabilities: list) -> list:
    """
    Returns the first moment, second moment, variance, and standard deviation.
    """
    results = []
    values = np.asarray(values, dtype = np.float64)
    probabilities = np.asarray(probabilities, dtype = np.float64)
    monent_1st = np.dot(values,probabilities)
    monent_2nd = np.dot(values**2,probabilities)
    variance = monent_2nd - monent_1st**2
    std = np.sqrt(variance)
    results.append(monent_1st)
    results.append(monent_2nd)
    results.append(variance)
    results.append(std)
    return results
