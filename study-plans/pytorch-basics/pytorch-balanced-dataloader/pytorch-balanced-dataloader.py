import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    dataset = TensorDataset(features,labels)
    class_counts = torch.bincount(labels)
    sample_weight = 1.0/class_counts[labels]
    sampler = WeightedRandomSampler(
        weights = sample_weight,
        num_samples = len(sample_weight),
        replacement = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        sampler = sampler
    )
    return dataloader
