import numpy as np

def auc_roc_trapezoidal(y_true: list, y_scores: list) -> dict:
    """
    Returns ROC coordinates and trapezoidal area.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    indices = np.argsort(-y_scores, kind = "stable")

    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0

    for i in indices:
        if y_true[i] == 1:
            tp+=1
        else:
            fp +=1
        fpr.append(fp/N)
        tpr.append(tp/P)

    auc = 0
    for i in range(1,n+1):
        auc += (fpr[i]-fpr[i-1])*(tpr[i] + tpr[i-1])/2
    return {"fpr": fpr, "tpr":tpr, "auc":auc}