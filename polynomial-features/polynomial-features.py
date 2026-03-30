def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    poly_features = []
    for x in values:
        poly_x = [x**p for p in range(degree +1)]
        poly_features.append(poly_x)
    return poly_features