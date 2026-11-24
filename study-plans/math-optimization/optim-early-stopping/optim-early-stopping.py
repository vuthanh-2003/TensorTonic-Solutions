def early_stopping(val_losses: list, patience: int) -> dict:
    """
    Returns the best and stopping epoch indices in a dictionary.
    """
    best_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return {"best_epoch": best_epoch, "stop_epoch": epoch}

    return {"best_epoch": best_epoch, "stop_epoch": len(val_losses) - 1}