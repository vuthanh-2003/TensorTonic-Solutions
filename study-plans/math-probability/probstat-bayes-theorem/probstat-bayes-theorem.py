def bayes_theorem(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """
    Returns the posterior probability rounded to four decimals.
    """
    p_b = p_a*p_b_given_a + (1-p_a)*p_b_given_not_a
    return round((p_b_given_a*p_a)/p_b,4)