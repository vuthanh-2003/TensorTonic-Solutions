from math import sqrt
from scipy.stats import norm

def ab_test(control_visitors: int, control_conversions: int, treatment_visitors: int, treatment_conversions: int, alpha: float) -> list:
    """
    Returns both rates, the z-statistic, p-value, and significance decision.
    """
    results = []
    p_c = control_conversions/control_visitors
    p_t = treatment_conversions/treatment_visitors
    p_mu = (control_conversions+treatment_conversions)/(control_visitors+treatment_visitors)
    z_stat = (p_t - p_c)/(sqrt(p_mu*(1-p_mu)*(1/control_visitors + 1/treatment_visitors)))
    p_value = 2*norm.sf(abs(z_stat))
    decide = p_value < alpha
    results.append(p_c)
    results.append(p_t)
    results.append(z_stat)
    results.append(p_value)
    results.append(decide)
    return results
    
