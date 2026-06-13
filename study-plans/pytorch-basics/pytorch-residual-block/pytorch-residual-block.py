import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Returns: None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(channels,channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        """
        Returns: output tensor
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = torch.relu(out)
        return out