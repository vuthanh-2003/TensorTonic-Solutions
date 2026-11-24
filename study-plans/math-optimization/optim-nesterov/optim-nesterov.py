import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns classical and Nesterov MSE loss curves in a dictionary.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    n,d = X.shape
    w_classic = np.zeros(d)
    w_nest = np.zeros(d)
    v_classic = np.zeros(d)
    v_nest = np.zeros(d)
    classical_losses = []
    nesterov_losses = []
    for _ in range(n_epochs):
        classic_loss = np.mean((X@w_classic-y)**2)
        nesterov_loss = np.mean((X@w_nest-y)**2)
        classical_losses.append(classic_loss)
        nesterov_losses.append(nesterov_loss)
        gradient_classic = (2/n)*X.T@(X@w_classic-y)
        w_look = w_nest - lr*beta*v_nest
        gradient_nest = (2/n)*X.T@(X@w_look-y)
        v_classic = beta*v_classic + gradient_classic
        v_nest = beta*v_nest + gradient_nest
        w_classic -= lr*v_classic
        w_nest -= lr*v_nest
    return {"classical_losses": classical_losses, "nesterov_losses": nesterov_losses}