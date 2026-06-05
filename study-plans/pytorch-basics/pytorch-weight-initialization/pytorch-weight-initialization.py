import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    if method == 'xavier_uniform':
        limit = torch.sqrt(torch.tensor(6/(fan_in+fan_out)))
        return torch.empty(fan_out,fan_in).uniform_(-limit,limit)
    elif method == 'he_uniform':
        limit = torch.sqrt(torch.tensor(6/fan_in))
        return torch.empty(fan_out,fan_in).uniform_(-limit,limit)
    elif method == 'xavier_normal':
        std = torch.sqrt(torch.tensor(2/(fan_in+fan_out)))
        return torch.empty(fan_out,fan_in).normal_(0,std)
    elif method == 'he_normal':
        std = torch.sqrt(torch.tensor(2/fan_in))
        return torch.empty(fan_out,fan_in).normal_(0,std)
    else:
        raise ValueError()
