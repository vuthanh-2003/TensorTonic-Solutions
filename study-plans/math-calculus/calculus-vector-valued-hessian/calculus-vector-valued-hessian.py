def vector_valued_hessian(x: float, y: float, z: float) -> list:
    """
    Returns the three Hessian matrices as a nested list.
    """
    h1 = [
        [float(2 * y), float(2 * x), 0.0],
        [float(2 * x), 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    h2 = [
        [0.0, 0.0, 0.0],
        [0.0, float(2 * z), float(2 * y)],
        [0.0, float(2 * y), 0.0],
    ]

    h3 = [
        [0.0, 0.0, float(2 * z)],
        [0.0, 0.0, 0.0],
        [float(2 * z), 0.0, float(2 * x)],
    ]

    return [h1, h2, h3]
