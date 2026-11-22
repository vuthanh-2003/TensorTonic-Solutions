def gradient_descent_nonconvex(w0: float, lr: float, n_iters: int) -> dict:
    """
    Returns critical points, classifications, and the full trajectory.
    """
    root = float(1.5**0.5)
    critical_points = [-root, 0.0, root]
    classifications = []
    for cp in critical_points:
        second_deriv = 12.0 * (cp**2) - 6.0
        if second_deriv > 0:
          classifications.append("local_min")
        elif second_deriv < 0:
          classifications.append("local_max")
        else:
          classifications.append("inconclusive")
    w = float(w0)
    lr_val = float(lr)
    trajectory = [w]
    for _ in range(n_iters):
        grad = 4.0 * (w**3) - 6.0 * w
        w = w - lr_val * grad
        trajectory.append(float(w))

    return {
      "critical_points": critical_points,
      "classifications": classifications,
      "trajectory": trajectory,
  }
