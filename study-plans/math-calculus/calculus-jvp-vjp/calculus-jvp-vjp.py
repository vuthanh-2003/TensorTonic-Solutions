import numpy as np

def jvp_vjp_2layer(W1: list, W2: list, x: list, x_dot: list, v_bar: list) -> dict:
    """
    Returns network output, JVP, VJP, and multiply counts.
    """
    W1 = np.array(W1)
    W2 = np.array(W2)
    x = np.array(x)
    x_dot = np.array(x_dot)
    v_bar = np.array(v_bar)
    m,d = W1.shape
    p = W2.shape[0]
    z = W1@x
    jvp_x = W1@x_dot
    f_z = np.maximum(z,0)
    jvp_z = jvp_x*(z>0)
    f = W2@f_z
    f_jvp = W2@jvp_z
    a_vjp = W2.T@v_bar
    z_bar = a_vjp*(z>0)
    x_bar = W1.T@z_bar
    count_both = m*(d+1+p)
    count_fulljvp = d*count_both
    count_fullvjp = p*count_both
    return {"f":f.tolist(), "jvp_result": f_jvp.tolist(), "vjp_result": x_bar.tolist(), "jvp_flops": count_both, "vjp_flops": count_both, "full_jacobian_jvp_flops": count_fulljvp, "full_jacobian_vjp_flops": count_fullvjp}
    