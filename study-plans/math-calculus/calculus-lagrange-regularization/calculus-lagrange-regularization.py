import numpy as np

def regularization_equivalence(A: list, b: list, lambda_val: float) -> dict:
    """
    Returns the unconstrained, penalized, and equivalent constrained results.
    """
    A = np.array(A)
    b = np.array(b)
    w_unreg = np.linalg.solve(A,b)
    def w_of(nu):
        I = np.eye(2)
        M = A + 2*nu*I
        return np.linalg.solve(M,b)
    w_reg = w_of(lambda_val)
    c = np.sum(w_reg**2)
    lo = 0.0
    hi = 1.0
    while np.sum(w_of(hi)**2) > c:
        hi = hi*2
    for i in range(100):
        mid = (lo+hi)/2
        if np.sum(w_of(mid)**2) > c:
            lo = mid
        else:
            hi = mid
    nu = mid
    w_con = w_of(nu)
    solution_match = bool(np.allclose(w_reg,w_con, atol = 1e-6))
    return {"constraint_bound": c, "lagrange_multiplier": nu, "solutions_match": solution_match, "w_constrained": w_con.tolist(), "w_regularized": w_reg.tolist(), "w_unregularized": w_unreg.tolist()}