import numpy as np

def gradient_check(f_name: str, x: float, analytical_deriv: float) -> dict:
    """
    Returns finite-difference errors and the best h for each method.
    """
    if f_name == "sin":
        f = np.sin
    elif f_name == "exp":
        f = np.exp
    elif f_name == "sigmoid":
        f = lambda val: 1/(1+np.exp(-val))
    elif f_name == "x_cubed":
        f = lambda val: val**3
    h_values = [10**(-i) for i in range(1,9)]
    forward_errors = []
    backward_errors = []
    central_errors = []
    for h in h_values:
        g_forward = (f(x+h) - f(x))/h
        forward_errors.append(abs(g_forward - analytical_deriv))

        g_backward = (f(x) - f(x-h))/h
        backward_errors.append(abs(g_backward - analytical_deriv))

        g_central = (f(x+h) - f(x-h))/(2*h)
        central_errors.append(abs(g_central-analytical_deriv))

        best_h_forward = h_values[np.argmin(forward_errors)]
        min_error_forward = min(forward_errors)

        best_h_backward = h_values[np.argmin(backward_errors)]
        min_error_backward = min(backward_errors)

        best_h_central = h_values[np.argmin(central_errors)]
        min_error_central = min(central_errors)

    return {
            "h_values": h_values,
            "forward_errors":forward_errors,
            "backward_errors": backward_errors,
            "central_errors":central_errors,
            "optimal_h_forward": best_h_forward,
            "optimal_h_backward": best_h_backward,
            "optimal_h_central": best_h_central
        }