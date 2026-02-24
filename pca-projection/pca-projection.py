import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean = np.mean(X, axis=0)
    X_c = X - mean
    C = np.dot(X_c.T, X_c) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    W = eigenvectors[:, top_k_indices]
    X_proj = np.dot(X_c, W)
    
    return X_proj