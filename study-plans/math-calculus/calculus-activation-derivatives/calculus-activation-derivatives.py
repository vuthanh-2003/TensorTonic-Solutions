import numpy as np

def activation_derivative(name: str, x: list) -> list:
    """
    Returns one analytical derivative for each input value.
    """
    x = np.array(x, dtype = np.float64)
    def sigmoid(value):
        return 1/(1+np.exp(-1*value))
    if name == "sigmoid":
        return (sigmoid(x)*(1-sigmoid(x))).flatten().tolist()
    elif name == "relu":
        relu_der = np.where(x>0,1.0,0.0).flatten().tolist()
        return relu_der
    elif name == "tanh":
        tanh = (np.exp(x) - np.exp(-1*x))/(np.exp(x) + np.exp(-1*x))
        return (1 - tanh**2).flatten().tolist()
    else:
        return (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))).flatten().tolist()
        
        