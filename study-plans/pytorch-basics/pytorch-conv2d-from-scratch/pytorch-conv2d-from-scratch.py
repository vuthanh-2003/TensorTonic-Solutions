import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()
        self.kernel_size = kernel_size 
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size,self.kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        batch_size, in_channels, height, width = x.shape
        out_h = height - self.kernel_size + 1
        out_w = width - self.kernel_size + 1
        output = torch.zeros(batch_size, self.out_channels, out_h, out_w)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        patch = x[b,:,h:h+self.kernel_size,w:w+self.kernel_size]
                        output[b,oc,h,w] = torch.sum(patch*self.weight[oc]) + self.bias[oc]
        return output
