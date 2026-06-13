import math

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """
    lr_track = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = base_lr*(step+1)/warmup_steps
            lr_track.append(lr)
        else:
            lr = base_lr*0.5*(1+math.cos(math.pi*(step-warmup_steps)/(total_steps-warmup_steps)))
            lr_track.append(lr)
    return lr_track