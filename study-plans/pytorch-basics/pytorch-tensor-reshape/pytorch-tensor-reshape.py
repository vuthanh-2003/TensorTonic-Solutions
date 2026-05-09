import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x).to(torch.float32)
    if op == 'transpose':
        return x.T.tolist()
    elif op == 'flatten':
        return torch.flatten(x, start_dim = 0).tolist()
    elif op == 'squeeze':
        return x.squeeze().tolist()
