import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype = torch.float32)
    if method == 'relu':
        output = torch.where(x>0,x,0).tolist()
    elif method == 'sigmoid':
        output = 1/(1+torch.exp(-x))
    elif method == 'tanh':
        output = (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
    elif method == 'leaky_relu':
        output = torch.clamp(x, min = 0) + 0.01*torch.clamp(x,max = 0)
    return output