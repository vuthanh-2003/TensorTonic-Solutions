def lr_stability_analysis(curvature: float, w0: float, n_iters: int) -> dict:
    """
    Returns the stability bound, learning rates, and three trajectories.
    """
    conv_bound = 2/curvature
    alpha_values = [0.5/curvature,1.9/curvature,2.1/curvature]
    trajectory = []
    for alpha in alpha_values:
        r = 1-alpha*curvature
        traject_list = []
        w_curr = w0
        traject_list.append(w0)
        for n in range(n_iters):
            w_curr = r*w_curr
            traject_list.append(w_curr)
        trajectory.append(traject_list)
    return {"alpha_values": alpha_values, "convergence_bound": conv_bound, "convergent_trajectory": trajectory[0], "divergent_trajectory": trajectory[2], "oscillatory_trajectory": trajectory[1]}

        
