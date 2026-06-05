import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    epoch_losses = []
    epoch_lrs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_lrs.append(current_lr)
        for x,y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss/len(dataloader))
        scheduler.step()
    return {"losses": epoch_losses, "lrs":epoch_lrs}
