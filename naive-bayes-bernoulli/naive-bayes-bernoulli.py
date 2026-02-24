import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    classes = np.unique(y_train)
    classes.sort()
    
    n_train, d = X_train.shape
    n_classes = len(classes)

    log_priors = np.zeros(n_classes)
    log_thetas = np.zeros((n_classes, d))
    log_one_minus_thetas = np.zeros((n_classes, d))

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_y = X_c.shape[0]
        
        log_priors[idx] = np.log(n_y / n_train)

        theta_y = (np.sum(X_c, axis=0) + 1) / (n_y + 2)

        log_thetas[idx] = np.log(theta_y)
        log_one_minus_thetas[idx] = np.log(1 - theta_y)
    
    term1 = np.dot(X_test, log_thetas.T)
    term2 = np.dot((1 - X_test), log_one_minus_thetas.T)
    
    log_posteriors = term1 + term2 + log_priors
    
    return log_posteriors