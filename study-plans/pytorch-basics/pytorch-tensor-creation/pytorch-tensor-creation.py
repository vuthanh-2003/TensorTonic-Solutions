import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == 'zeros':
        output = torch.zeros(tuple(shape)).tolist()
    elif method == 'ones':
        output = torch.ones(tuple(shape)).tolist()
    elif method == 'full':
        output = torch.full(tuple(shape), value).tolist()
    return output
    