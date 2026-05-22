import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if not self.training:
            return x
        if self.p == 0:
            return x
        if self.p == 1:
            return torch.zeros_like(x)
        mask = torch.rand_like(x) > self.p
        return (x*mask)/(1-self.p)
