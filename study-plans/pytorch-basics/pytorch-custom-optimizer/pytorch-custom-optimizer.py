import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    Returns: loss or None from step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = dict(lr = lr,momentum = momentum)
        super().__init__(params,defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if momentum > 0:
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad
                param.data.add_(-lr*update)
        return loss
