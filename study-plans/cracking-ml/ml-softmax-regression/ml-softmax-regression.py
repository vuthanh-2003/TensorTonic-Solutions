import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.array(X,dtype = np.float64)
    y = np.array(y,dtype = np.int64)
    n,d = X.shape
    weights = np.zeros((d, num_classes))
    bias = np.zeros((1,num_classes))
    y_1hot = np.eye(num_classes)[y]
    for i in range(n_iters):
        logits = X@weights + bias
        exp = np.exp(logits - logits.max(axis = 1, keepdims = True))
        prob = exp/np.sum(exp, axis = 1, keepdims = True)
        error = prob - y_1hot
        grad_w = (X.T@error)/n
        grad_b = np.mean(error, axis = 0)
        weights -= lr*grad_w
        bias -= lr*grad_b
    return (np.round(weights,4).tolist(), np.round(bias,4).flatten().tolist())
    
