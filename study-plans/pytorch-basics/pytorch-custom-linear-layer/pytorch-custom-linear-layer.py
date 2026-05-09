import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight)
        bound = 1/math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound,bound)

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        z = x@self.weight.t()+ self.bias
        return z
