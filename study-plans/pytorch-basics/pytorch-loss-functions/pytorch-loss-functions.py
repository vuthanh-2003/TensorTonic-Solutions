import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype = torch.float32)
    target = torch.tensor(target, dtype = torch.float32)
    if method == 'mse':
        L = ((pred - target)**2).mean()
    elif method == 'cross_entropy':
        log_probs = torch.log_softmax(pred, dim=1)
        selected = log_probs[range(pred.shape[0]),target.long()]
        L = -selected.mean()
    elif method == 'huber':
        a = (pred - target)
        L = torch.where(torch.abs(a) <= delta, (1/2)*(a**2), delta*(torch.abs(a) - 
                                                                    (1/2)*delta)).mean()
    return L