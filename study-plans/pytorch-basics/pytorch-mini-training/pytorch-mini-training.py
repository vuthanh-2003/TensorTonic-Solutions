import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    loss_total = 0
    for x,y in dataloader:
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(y,pred)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total/len(dataloader)
