import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    seq_q = Q.shape[1]
    seq_k = K.shape[1]
    d_k = Q.shape[2]
    mask = torch.triu(torch.ones(seq_q,seq_k), diagonal = 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    output = torch.softmax((Q@K.transpose(1,2)/d_k**0.5)+mask, dim = -1)@V
    return output