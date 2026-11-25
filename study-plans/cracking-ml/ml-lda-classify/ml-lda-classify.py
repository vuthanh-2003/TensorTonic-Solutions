import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.asarray(X_train, dtype = np.float64)
    X_test = np.asarray(X_test, dtype = np.float64)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    K = len(classes)
    n,d = X_train.shape
    muy_c = []
    p_c = []
    s = np.zeros((d,d))
    for c in classes:
        mask = y_train == c
        muy_c.append(X_train[mask].mean(axis = 0))
        p_c.append(len(X_train[mask]))
        centered = X_train[mask] - muy_c[-1]
        s += centered.T@centered
    muy_c = np.asarray(muy_c)
    p_c = np.asarray(p_c)/n
    sigma = (s/(n-K)) + 1e-6*np.eye(d)
    w_c = np.linalg.solve(sigma,muy_c.T).T
    z =X_test@w_c.T -(1/2)*np.sum(muy_c*w_c,axis = 1) + np.log(p_c)
    pred = classes[np.argmax(z,axis = 1)]
    result = [int(m) for m in pred]
    return result
