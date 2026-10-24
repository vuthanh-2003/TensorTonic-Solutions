import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype = np.float64)
    y = np.array(y, dtype = np.float64).reshape(-1,1)
    n,d = X.shape
    weights = np.zeros((d,1))
    bias = 0.0
    for epoch in range(epochs):
        y_pred = X@weights + bias
        error = y_pred - y
        dw = (2.0/n)*(X.T@error) + alpha*np.sign(weights)
        db = (2.0/n)*np.sum(error)
        weights -= lr*dw
        bias -= lr*db
    weight_list = [round(float(w), 4) for w in weights.flatten()]
    bias_val = round(float(bias), 4)
    
    return weight_list, bias_val
