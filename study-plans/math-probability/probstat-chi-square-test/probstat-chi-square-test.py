import numpy as np
from scipy.stats import chi2

def chi_square_test(observed: list, expected: list) -> list:
    """
    Returns the chi-square statistic, degrees of freedom, p-value, and decision.
    """
    chi_2 = sum([((o - e)**2)/e for o,e in zip(observed,expected)])
    df = len(observed) - 1
    p_value = chi2.sf(chi_2, df)
    decide = p_value < 0.05
    return [chi_2, df,p_value,decide]
