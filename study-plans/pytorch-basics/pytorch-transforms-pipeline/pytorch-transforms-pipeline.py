import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, image):
        image = image/255.0
        image = image.permute(2,0,1)
        self.mean = self.mean.view(-1,1,1)
        self.std = self.std.view(-1,1,1)
        image = (image - self.mean)/self.std
        return image
