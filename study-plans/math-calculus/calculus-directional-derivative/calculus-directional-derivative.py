import numpy as np

def directional_derivative_analysis(w: list, v: list) -> dict:
    """
    Returns gradient and directional steepest-descent data.
    """
    w = np.asarray(w)
    v = np.asarray(v)
    gradient = np.asarray([2*w[0], 6*w[1]])
    direct_deriv = np.dot(gradient,v)
    max_descent_rate = -np.linalg.norm(gradient)
    steepest_descent = -gradient/(-max_descent_rate) if max_descent_rate!=0 else np.zeros(2)
    return {"gradient": gradient.tolist(), "directional_derivative": direct_deriv, "steepest_descent_direction": steepest_descent.tolist(), "max_descent_rate":max_descent_rate}
    