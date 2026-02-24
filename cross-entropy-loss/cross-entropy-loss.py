import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    
    N = y_true.shape[0]
    correct_class_probs = y_pred[np.arange(N), y_true]

    loss = -np.mean(np.log(correct_class_probs))
    
    return loss