def independence_test(p_a: float, p_b: float, p_a_and_b: float) -> dict:
    """
    Returns the rounded product and independence decision in a dictionary.
    """
    p_a_times_p_b = p_a*p_b
    if abs(p_a_times_p_b-p_a_and_b) < 1e-10:
        is_independent = True
    else:
        is_independent = False
    return {"p_a_times_p_b": p_a_times_p_b, "is_independent": is_independent}