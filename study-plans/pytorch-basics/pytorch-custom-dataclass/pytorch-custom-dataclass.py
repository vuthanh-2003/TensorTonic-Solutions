import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        data = torch.tensor(data, dtype = torch.float32)
        mask = torch.ones(data.shape[1], dtype = torch.bool)
        mask[label_col] = False
        self.x = data[:,mask]
        self.y = data[:,label_col].unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
