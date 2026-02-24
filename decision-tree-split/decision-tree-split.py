import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data
    by minimizing the weighted Gini impurity.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    
    n_samples, n_features = X.shape
    
    def _gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)
    
    best_feature = -1
    best_threshold = None
    min_gini_split = float('inf')
    
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        sorted_unique_values = np.unique(feature_values)

        if len(sorted_unique_values) <= 1:
            continue
            
        thresholds = (sorted_unique_values[:-1] + sorted_unique_values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            n_left = len(y_left)
            n_right = len(y_right)
            
            gini_left = _gini(y_left)
            gini_right = _gini(y_right)
            
            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

            if weighted_gini < min_gini_split:
                min_gini_split = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold
                
    return [best_feature, best_threshold]