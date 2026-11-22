import numpy as np

def taylor_approximation(w0: float, n_points: int = 100) -> dict:
    """
    Returns the sampled loss and its first two Taylor approximations.
    """
    L = np.sin(w0) + 0.1*w0**2
    dL_w0 = np.cos(w0) + 0.2*w0
    d2L_w0 = -np.sin(w0) + 0.2
    w = np.linspace(w0-2,w0+2,n_points)
    d = w - w0
    true_loss = np.sin(w) + 0.1*w**2
    linear_approx = L + dL_w0*d
    quadratic_approx = linear_approx + 0.5 * d2L_w0 * d**2
    return {
      "L_at_w0": L,
      "dL_at_w0": dL_w0,
        "d2L_at_w0": d2L_w0,
      "w_values": w.tolist(),
      "true_loss": true_loss.tolist(),
      "linear_approx": linear_approx.tolist(),
      "quadratic_approx": quadratic_approx.tolist(),
  }
