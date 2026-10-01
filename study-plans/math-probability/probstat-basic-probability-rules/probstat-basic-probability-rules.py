def basic_probability(p_a: float, p_b: float, p_a_and_b: float) -> list:
    """
    Returns four rounded probability values in the required order.
    """
    results = []
    results.append(p_a + p_b - p_a_and_b)
    results.append(1-p_a)
    results.append(1-p_b)
    results.append(p_a - p_a_and_b)
    return results