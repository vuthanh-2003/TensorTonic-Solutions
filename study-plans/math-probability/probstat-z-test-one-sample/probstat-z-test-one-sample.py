from math import sqrt
from scipy.stats import norm

def z_test_one_sample(x_bar: float, mu_0: float, sigma: float, n: int, alpha: float) -> list:
    """
    Returns the z-statistic, two-sided p-value, and rejection decision.
    """
    results = []
    z = (x_bar - mu_0)/(sigma/sqrt(n))
    p_value = 2*norm.cdf(-1*abs(z))
    if p_value - alpha < 0:
        decision = True
    else:
        decision = False
    results.append(z)
    results.append(p_value)
    results.append(decision)
    return results
    
