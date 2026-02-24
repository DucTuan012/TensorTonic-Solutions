import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    is_single_point = (points.ndim == 1)
    points_2d = np.atleast_2d(points)
    
    N = points_2d.shape[0]
    
    ones = np.ones((N, 1))
    p_h = np.hstack((points_2d, ones))  
    
    p_h_prime = np.dot(p_h, T.T)
    
    p_prime = p_h_prime[:, :3]
    
    if is_single_point:
        return p_prime[0]
        
    return p_prime