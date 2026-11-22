import numpy as np

def second_order_partials(x: float, y: float) -> dict:
    """
    Returns the two pure second-order partial derivatives.
    """
    return {"d2f_dx2": -(y**2)*np.sin(x*y), "d2f_dy2": -(x**2)*np.sin(x*y)}