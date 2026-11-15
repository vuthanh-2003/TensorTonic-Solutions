import numpy as np

def jacobian_network_layer(W: list, b: list, x: list) -> dict:
    """
    Returns layer values and analytical and numerical Jacobians.
    """
    W = np.array(W,dtype = np.float64)
    b = np.array(b,dtype = np.float64)
    x = np.array(x,dtype = np.float64)
    h = 1e-5
    def forward(v):
        return 1/(1+np.exp(-v))
    z = W@x + b
    f = forward(z)
    f_grad = f*(1-f)
    J_analytic = W*f_grad.reshape(W.shape[0],-1)
    J_num = np.zeros_like(W)
    for j in range(J_num.shape[1]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        z_plus = W@x_plus + b
        z_minus = W@x_minus + b
        f_plus = forward(z_plus)
        f_minus = forward(z_minus)
        J_num[:,j] = (f_plus-f_minus)/(2*h)
    return {"z":z.tolist(), "f": f.tolist(), "analytical_jacobian": J_analytic.tolist(), "numerical_jacobian": J_num.tolist()}
    
    