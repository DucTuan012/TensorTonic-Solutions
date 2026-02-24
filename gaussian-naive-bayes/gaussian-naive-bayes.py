import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    
    classes = np.unique(y_train)
    classes.sort()
    
    n_classes = len(classes)
    n_samples, n_features = X_train.shape
    
    log_priors = np.zeros(n_classes)
    means = np.zeros((n_classes, n_features))
    variances = np.zeros((n_classes, n_features))
    
    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        log_priors[idx] = np.log(X_c.shape[0] / n_samples)
        
        means[idx, :] = np.mean(X_c, axis=0)
        variances[idx, :] = np.var(X_c, axis=0)
        
    epsilon = 1e-9
    variances += epsilon
    
    n_test = X_test.shape[0]
    log_posteriors = np.zeros((n_test, n_classes))
    
    for idx, c in enumerate(classes):
        mean = means[idx, :]
        var = variances[idx, :]
        
        term1 = -0.5 * np.log(2 * np.pi * var)
        term2 = -((X_test - mean) ** 2) / (2 * var)
        
        log_likelihood = np.sum(term1 + term2, axis=1)
        log_posteriors[:, idx] = log_likelihood + log_priors[idx]
        
    predictions = classes[np.argmax(log_posteriors, axis=1)]

    return predictions.tolist()