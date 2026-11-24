import numpy as np

def momentum_gd(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns vanilla and momentum MSE loss curves in a dictionary.
    """
    X_arr = np.array(X,dtype = np.float64)
    y_arr = np.array(y,dtype = np.float64)
    n,d = X_arr.shape
    w_van = np.zeros(d)
    w_mom = np.zeros(d)
    v = np.zeros(d)
    van_losses = []
    moment_losses = []
    for _ in range(n_epochs):
        van_loss = float(np.mean((X_arr@w_van - y_arr)**2))
        moment_loss = float(np.mean((X_arr@w_mom - y_arr)**2))
        van_losses.append(van_loss)
        moment_losses.append(moment_loss)
        gradient_van = (2/n)*X_arr.T@(X_arr@w_van-y_arr)
        gradient_mom = (2/n)*X_arr.T@(X_arr@w_mom-y_arr)
        v = beta*v + gradient_mom
        w_van -= lr*gradient_van
        w_mom -= lr*v
    return {"vanilla_losses": van_losses, "momentum_losses": moment_losses}