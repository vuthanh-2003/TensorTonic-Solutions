def matrix_polynomial_gradient(x: float) -> list:
    """
    Returns the 3 by 3 entrywise derivative matrix.
    """
    result = []
    for i in range(1, 4):
        row = []
        for j in range(1, 4):
            power = i + j - 1
            deriv = float(power * (x ** (power - 1)))
            row.append(deriv)
        result.append(row)
    return result