import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    """
    Returns a dictionary: weights (float list), bias (float), predictions (integer list).
    """
    X = np.array(X,dtype = np.float64)
    y = np.array(y,dtype = np.float64).reshape(-1,1)
    n,d = X.shape
    w = np.zeros(d)
    b = 0.0
    s_w = np.zeros(d)
    s_b = 0.0
    c= 1
    T = n*epochs
    for epoch in range(epochs):
        for i in range(n):
            xi = X[i,:]
            yi = y[i,:]
            margin = yi*(np.dot(w,xi)+b)
            if margin <= 0:
                s_w += c*w
                s_b += c*b

                w += lr*yi*xi
                b += lr*yi

                c = 1
            else:
                c+=1
    s_w+=c*w
    s_b+=c*b
    w_avg=s_w/T
    b_avg=s_b/T
    margin = X@w_avg+b_avg
    y_pred = np.where(margin > 0, 1, -1)
    return {"weights": [round(float(val), 4) for val in w_avg.flatten()],"bias": round(float(b_avg.tolist()[0]),4), "predictions": y_pred.flatten().tolist()}
            
