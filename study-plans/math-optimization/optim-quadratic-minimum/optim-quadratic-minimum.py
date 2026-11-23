def quadratic_minimum(a: float, b: float, c: float) -> dict:
    """
    Returns the minimizer and minimum value in a dictionary.
    """
    x_star = -b/(2*a)
    f_min = a*(x_star)**2 +b*x_star +c
    return {"x_star": round(float(x_star),6), "f_min": round(float(f_min),6)}