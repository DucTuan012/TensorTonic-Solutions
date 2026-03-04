import numpy as np

def matrix_inverse(A):
    # Ép kiểu list mặc định của Python sang NumPy array
    A_np = np.array(A)
    
    # Đảm bảo input thực sự là ma trận 2D
    if len(A_np.shape) != 2:
        return None
        
    rows, cols = A_np.shape
    
    # 1. Kiểm tra ma trận vuông
    if rows != cols:
        return None
        
    # 2 & 3. Tính nghịch đảo và xử lý ma trận suy biến
    try:
        A_inv = np.linalg.inv(A_np)
        # Bỏ .tolist() đi, trả về trực tiếp numpy array như hệ thống yêu cầu!
        return A_inv
        
    except np.linalg.LinAlgError:
        return None