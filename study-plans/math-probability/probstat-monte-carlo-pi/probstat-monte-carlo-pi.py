import numpy as np

def monte_carlo_pi(n_points: int, seed: int) -> dict:
    """
    Returns the pi estimate and its absolute error.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n_points,2))
    inside = (pts**2).sum(axis = 1) <= 1
    estimate = 4*inside.mean()
    error = abs(estimate-np.pi)
    return {"error": round(float(error),4), "estimate": round(float(estimate),4)}
