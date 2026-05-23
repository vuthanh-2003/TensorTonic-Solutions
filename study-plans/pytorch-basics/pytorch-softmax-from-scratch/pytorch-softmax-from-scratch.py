import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    logits = torch.tensor(logits,dtype = torch.float32)
    shifted = logits - torch.max(logits,dim = 1, keepdim = True)[0]
    exps = torch.exp(shifted)
    return exps/exps.sum(dim=1,keepdim = True)
