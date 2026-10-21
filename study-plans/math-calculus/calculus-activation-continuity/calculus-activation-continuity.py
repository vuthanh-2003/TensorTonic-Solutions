import numpy as np

def activation_continuity_analysis(x: list) -> dict:
    """
    Returns nondifferentiable tested points for three activations.
    """
    arr = np.array(x, dtype=float)
    h = 1e-7
    tol = 1e-5
    
    # 1. ReLU
    relu_f = np.maximum(0.0, arr)
    relu_left = (relu_f - np.maximum(0.0, arr - h)) / h
    relu_right = (np.maximum(0.0, arr + h) - relu_f) / h
    relu_mask = np.abs(relu_left - relu_right) >= tol
    relu_result = arr[relu_mask].tolist()
    
    # 2. Leaky ReLU (với hệ số alpha = 0.01)
    alpha = 0.01
    lrelu_f = np.where(arr >= 0, arr, alpha * arr)
    arr_minus = arr - h
    arr_plus = arr + h
    lrelu_minus = np.where(arr_minus >= 0, arr_minus, alpha * arr_minus)
    lrelu_plus = np.where(arr_plus >= 0, arr_plus, alpha * arr_plus)
    
    lrelu_left = (lrelu_f - lrelu_minus) / h
    lrelu_right = (lrelu_plus - lrelu_f) / h
    lrelu_mask = np.abs(lrelu_left - lrelu_right) >= tol
    lrelu_result = arr[lrelu_mask].tolist()
    
    # 3. GELU (Sử dụng công thức xấp xỉ tanh)
    def gelu(v):
        return 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * (v ** 3))))
    
    gelu_f = gelu(arr)
    gelu_left = (gelu_f - gelu(arr - h)) / h
    gelu_right = (gelu(arr + h) - gelu_f) / h
    gelu_mask = np.abs(gelu_left - gelu_right) >= tol
    gelu_result = arr[gelu_mask].tolist()
    
    return {
        "relu": relu_result,
        "leaky_relu": lrelu_result,
        "gelu": gelu_result
    }