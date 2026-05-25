import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w_init = torch.tensor(
        w_init,
        dtype=torch.float32,
        requires_grad=True
    )

    for idx, (x, y) in enumerate(micro_batches):

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        loss = ((w_init @ x.T - y) ** 2).mean()

        loss.backward()

        # đủ accum_steps thì update
        if (idx + 1) % accum_steps == 0:

            with torch.no_grad():

                gradient = w_init.grad / accum_steps

                w_init -= lr * gradient

            w_init.grad.zero_()

    return (w_init, gradient)
