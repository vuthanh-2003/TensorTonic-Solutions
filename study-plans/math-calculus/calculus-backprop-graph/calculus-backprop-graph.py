def backprop_graph(w1: float, w2: float, x: float) -> dict:
    """
    Returns the forward intermediates and leaf gradients.
    """
    x_sq = x * x
    prod1 = w1 * x
    prod2 = w2 * x_sq
    sum_val = prod1 + prod2
    output = sum_val if sum_val > 0.0 else 0.0
    d_output = 1.0
    relu_grad = 1.0 if sum_val > 0.0 else 0.0
    d_sum_val = d_output * relu_grad
    d_prod1 = d_sum_val
    d_prod2 = d_sum_val
    d_w1 = d_prod1 * x
    d_x = d_prod1 * w1
    d_w2 = d_prod2 * x_sq
    d_x_sq = d_prod2 * w2
    d_x += d_x_sq * (2.0 * x)
    return {
      "forward": {
          "x_sq": float(x_sq),
          "prod1": float(prod1),
          "prod2": float(prod2),
          "sum_val": float(sum_val),
          "output": float(output),
      },
      "gradients": {
          "d_w1": float(d_w1),
          "d_w2": float(d_w2),
          "d_x": float(d_x),
      },
  }
