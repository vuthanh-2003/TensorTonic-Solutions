import numpy as np
from scipy.stats import t

def correlation_analysis(x: list, y: list) -> dict:
    """
    Returns correlation, explained proportion, test statistic, p-value, and significance.
    """
    x = np.array(x,dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    n=len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    r = np.dot((x-x_mean), (y-y_mean))/(np.sqrt(np.dot((x-x_mean),(x-x_mean)))*np.sqrt(np.dot((y-y_mean),(y-y_mean))))
    r_squared = r**2
    t_stat = r*((n-2)/(1-r**2))**0.5
    p_value = 2*(1-t.cdf(abs(t_stat),n-2))
    sig = p_value < 0.05
    return {"p_value": p_value, "r": r, "r_squared": r_squared, "significant": sig, "t_stat": t_stat}
