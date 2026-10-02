def conditional_probability(p_a: float, p_b: float, p_a_and_b: float) -> list:
    """
    Returns both rounded conditional probabilities in the required order.
    """
    results = []
    results.append(round(p_a_and_b/p_b,4))
    results.append(round(p_a_and_b/p_a,4))
    return results