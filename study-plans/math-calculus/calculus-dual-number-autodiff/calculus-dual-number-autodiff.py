import math

def dual_number_autodiff(x_val: float) -> dict:
    """
    Returns the function value, two derivatives, and their comparison.
    """
    def dual_add(u,v):
        return (u[0] + v[0], u[1] + v[1])
    def dual_mul(u,v):
        return (u[0]*v[0], u[0]*v[1] + u[1]*v[0])
    def dual_sin(u):
        return (math.sin(u[0]), u[1]*math.cos(u[0]))
    def dual_exp(u):
        exp_val = math.exp(u[0])
        return (exp_val, u[1] * exp_val)
    x = (float(x_val), 1.0)
    x_sq = dual_mul(x, x)
    sin_x = dual_sin(x)
    term1 = dual_mul(x_sq, sin_x)
    term2 = dual_exp(x)
    f_dual = dual_add(term1, term2)
    f_val, f_deriv_dual = f_dual
    f_deriv_analytical = (
      2 * x_val * math.sin(x_val)
      + (x_val**2) * math.cos(x_val)
      + math.exp(x_val)
  )
    match = bool(math.isclose(f_deriv_dual, f_deriv_analytical))
    return {
      "f_val": float(f_val),
      "f_deriv_dual": float(f_deriv_dual),
      "f_deriv_analytical": float(f_deriv_analytical),
      "match": match,
  }
