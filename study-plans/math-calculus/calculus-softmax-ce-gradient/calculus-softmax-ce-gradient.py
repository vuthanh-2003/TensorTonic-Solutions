import numpy as np

def softmax_cross_entropy_gradient(logits: list, labels: list) -> dict:
    """
    Returns probabilities, loss, two gradients, and their comparison.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    p = exp_z / np.sum(exp_z)
    loss = float(-np.sum(y * np.log(p)))
    n = len(p)
    jacobian = np.diag(p) - np.outer(p, p)
    dL_dp = -y / p
    grad_chain_rule = dL_dp @ jacobian
    grad_direct = p - y
    match = bool(np.allclose(grad_chain_rule, grad_direct, atol=1e-7, rtol=0.0))
    return {
      "softmax": p.tolist(),
      "loss": loss,
      "grad_chain_rule": grad_chain_rule.tolist(),
      "grad_direct": grad_direct.tolist(),
      "match": match,
  }
