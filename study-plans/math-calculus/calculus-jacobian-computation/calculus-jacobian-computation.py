def jacobian_3d(x: float, y: float, z: float) -> list:
    """
    Returns the 3 by 3 Jacobian as a nested list.
    """
    return [
        [float(2 * x * y), float(x * x), 0.0],
        [0.0, float(2 * y), 1.0],
        [float(z * z), 0.0, float(2 * x * z)],
    ]