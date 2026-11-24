def vanilla_gradient_descent(x0: float, y0: float, lr: float, n_iters: int) -> dict:
    """
    Returns the trajectory, final point, and final function value.
    """
    trajectory = [[x0,y0]]
    for n in range(n_iters):
        x_new = trajectory[n][0] - 2*lr*trajectory[n][0]
        y_new = trajectory[n][1] - 6*lr*trajectory[n][1]
        trajectory.append([x_new,y_new])
    final_value = trajectory[-1][0]**2 + 3*trajectory[-1][1]**2
    return {"trajectory": trajectory, "final_point": trajectory[-1], "final_value": final_value}