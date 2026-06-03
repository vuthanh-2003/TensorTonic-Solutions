import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    train_losses = []
    val_losses = []
    best_val = float('inf')
    stopped_epoch = 0
    patient = 0
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_loader))
        model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                pred = model(x)
                loss = criterion(pred,y)
                val_loss += loss.item()
            val_losses.append(val_loss/len(val_loader))
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            patient = 0
        else:
            patient += 1
        if patient == patience:
            stopped_epoch = epoch +1
            break
        if stopped_epoch == 0:
            stopped_epoch = max_epochs
    return {"train_losses": train_losses, 'val_losses': val_losses, 'stopped_epoch': int(stopped_epoch)}
        
                
