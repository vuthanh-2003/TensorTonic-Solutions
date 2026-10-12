from scipy.stats import norm, t

def p_value_analysis(test_stat: float, dist_type: str, df: int, tail: str) -> dict:
    """
    Returns the p-value and significance classifications.
    """
    if dist_type == 'z':
        dist = norm
    else:
        dist = t(df = df)
    if tail == "left":
        p_value = dist.cdf(test_stat)
    elif tail == "right":
        p_value = dist.sf(test_stat)
    else:
        p_value = 2*dist.sf(abs(test_stat))

    significant_at_01 = p_value < 0.01
    significant_at_05 = p_value < 0.05
    significant_at_10 = p_value < 0.10
    return {"p_value": p_value, "significant_at_01":significant_at_01, "significant_at_05":significant_at_05, "significant_at_10":significant_at_10}
