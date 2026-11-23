def paraboloid_minimum(a: float, b: float, c: float, d: float, e: float) -> dict:
    """
    Returns the minimizer coordinates and minimum value in a dictionary.
    """
    x_star = -c/(2*a)
    y_star = -d/(2*b)
    f_min = a*x_star**2 + b*y_star**2 +c*x_star + d*y_star +e
    return {"x_star": round(float(x_star),6), "y_star": round(float(y_star),6), "f_min": round(float(f_min),6)}