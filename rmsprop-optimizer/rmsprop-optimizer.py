import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)
    gradient = g*g
    new_s = beta*s + (1-beta)*gradient
    new_w = w - (lr/(np.sqrt((new_s + eps))))*g
    return new_w, new_s