def lr_schedule_analysis(alpha_0: float, k: float) -> dict:
    """
    Returns the schedule limit and two convergence results.
    """
    if alpha_0 == 0.0:
        return {
            "limit": 0.0,
            "sum_diverges": False,
            "sum_sq_converges": True
        }
    
    if k == 0.0:
        return {
            "limit": float(alpha_0),
            "sum_diverges": True,
            "sum_sq_converges": False
        }
    return {
        "limit": 0.0,
        "sum_diverges": True,
        "sum_sq_converges": True
    }