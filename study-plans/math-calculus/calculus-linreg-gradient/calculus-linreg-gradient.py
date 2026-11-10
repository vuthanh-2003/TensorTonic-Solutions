import numpy as np

def linear_regression_gradient(X: list, y: list, w: list) -> dict:
    """
    Returns the SSE loss and analytical and numerical gradients.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w)
    def loss(weight):
        return np.dot(X@weight-y,X@weight-y)
    sse = loss(w)
    a_grad = 2*X.T@(X@w - y)
    h = 1e-5
    numerical_grad = []
    for i in range(len(w)):
        e = np.eye(w.shape[0])[i]
        w_plus = w + e*h
        w_minus = w - e*h
        numerical_grad.append((loss(w_plus) - loss(w_minus))/(2*h))
    return {"loss": sse, "analytical_gradient": a_grad.tolist(), "numerical_gradient": numerical_grad}