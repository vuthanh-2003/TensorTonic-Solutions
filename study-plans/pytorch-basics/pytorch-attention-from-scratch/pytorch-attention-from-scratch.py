import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    d_k = torch.tensor(Q.shape[-1], dtype = torch.float32)
    scores = Q@K.transpose(-2,-1)/torch.sqrt(d_k)
    weights = torch.softmax(scores, dim = -1)
    return weights@V