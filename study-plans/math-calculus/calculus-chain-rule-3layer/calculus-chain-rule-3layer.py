import numpy as np

def chain_rule_3layer(w1: float, w2: float, w3: float, x: float) -> dict:
    """
    Returns chain-rule factors and analytical and numerical gradients.
    """
    def sigmoid(y):
        return 1/(1+np.exp(-y))
    z1 = w1*x
    a1 = sigmoid(z1)
    z2 = w2*a1
    a2 = sigmoid(z2)
    z3 = w3*a2
    a3 = sigmoid(z3)
    factors = [a3*(1-a3),w3,a2*(1-a2),w2,a1*(1-a1),x]
    gradient = np.prod(factors)
    h = 1e-5
    w1_plus = w1+h
    w1_minus = w1-h
    a3_plus = sigmoid(w3*sigmoid(w2*sigmoid(w1_plus*x)))
    a3_minus = sigmoid(w3*sigmoid(w2*sigmoid(w1_minus*x)))
    numerical_grad = (a3_plus-a3_minus)/(2*h)
    return {"factors": factors, "analytical_gradient": gradient, "numerical_gradient": numerical_grad}