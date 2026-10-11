import numpy as np
from scipy.stats import t

def t_test_one_sample(data: list, mu_0: float, alpha: float) -> dict:
    """
    Returns the t-statistic, degrees of freedom, p-value, and rejection decision.
    """
    n = len(data)
    df = n - 1
    x_bar = sum(data)/n
    std = (sum([(x_i - x_bar)**2 for x_i in data])/(n-1))**0.5
    t_stat = (x_bar - mu_0)/(std/((n)**0.5))
    p_value = 2*t.sf(abs(t_stat), df = df)
    if p_value - alpha > 0:
        reject_null = False
    else:
        reject_null = True
    return {"degrees_of_freedom": df, "p_value": p_value, "reject_null": reject_null, "t_statistic": t_stat}
    
