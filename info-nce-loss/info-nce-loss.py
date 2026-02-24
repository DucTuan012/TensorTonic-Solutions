import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    S = np.dot(Z1, Z2.T) / temperature
    S_max = np.max(S, axis=1, keepdims=True)
    S_shifted = S - S_max
    
    log_sum_exp = np.log(np.sum(np.exp(S_shifted), axis=1)) + S_max.flatten()

    S_positives = np.diagonal(S)

    loss = np.mean(log_sum_exp - S_positives)
    
    return loss