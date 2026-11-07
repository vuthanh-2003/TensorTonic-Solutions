import numpy as np

def expected_value_x_squared(mu: float, sigma: float, n: int) -> dict:
    """
    Returns trapezoidal, Simpson, and analytical second moments.
    """
    a = mu - 5*sigma
    b = mu + 5*sigma
    x = np.linspace(a,b,num = n+1)
    analytical = mu**2 + sigma**2
    p_x = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)
    h = (b-a)/n
    weight_trape = np.full(x.shape, 2)
    weight_trape[0] = 1
    weight_trape[-1] = 1
    weight_simpson = np.array([2 if i%2 == 0 else 4 for i in range(len(x))])
    weight_simpson[0] = 1
    weight_simpson[-1] = 1
    g_x = (x**2)*p_x
    trapezoidal = (h/2)*np.dot(weight_trape,g_x)
    simpson = (h/3)*np.dot(weight_simpson,g_x)
    return {"trapezoidal": trapezoidal, "simpsons": simpson, "analytical":analytical}
    
    
    