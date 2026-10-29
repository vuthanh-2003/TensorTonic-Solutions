import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.array(X_train, dtype = np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype = np.float64)
    classes = np.unique(y_train)
    n = len(y_train)

    priors = []
    means = []
    vars = []
    for c in classes:
        X_c = X_train[y_train == c]
        priors.append(len(X_c)/n)
        means.append(np.mean(X_c, axis = 0))
        vars.append(np.var(X_c, axis = 0, ddof = 0))
    priors = np.array(priors)
    means = np.array(means)
    vars = np.array(vars)
    eps = 1e-9
    vars = vars + eps
    pred = []
    for x in X_test:
        log_prior = np.log(priors)
        log_likelihood = -0.5*np.log(2*np.pi*vars) - ((x-means)**2)/(2*vars)
        log_likelihood_sum  = np.sum(log_likelihood,axis = 1)
        post_score = log_prior + log_likelihood_sum
        pred_class = classes[np.argmax(post_score)]
        pred.append(int(pred_class))
    return pred