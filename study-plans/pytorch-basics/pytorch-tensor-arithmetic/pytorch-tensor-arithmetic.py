import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    if op == 'add':
        output = torch.add(x,y).tolist()
    elif op == 'multiply':
        output = torch.multiply(x,y).tolist()
    elif op == 'matmul':
        output = (x@y).tolist()
    elif op == 'power':
        output = (x**y).tolist()
    elif op == 'max':
        output = torch.maximum(x,y).tolist()
    return output