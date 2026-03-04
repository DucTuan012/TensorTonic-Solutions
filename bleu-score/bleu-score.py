import math
from collections import Counter

def get_ngrams(tokens, n):
    """Hàm phụ trợ: Tạo n-grams từ danh sách các token."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(candidate, reference, max_n):
    c = len(candidate)
    r = len(reference)
    
    # Xử lý ngoại lệ: Nếu candidate rỗng, điểm BLEU = 0
    if c == 0:
        return 0.0
        
    precisions = []
    
    # 1. Tính modified precision (p_n) cho từng n-gram từ 1 đến max_n
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        
        # Nếu candidate ngắn hơn n, không thể tạo n-gram => precision = 0
        if not cand_ngrams:
            return 0.0 
            
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # Tử số: Tổng các min(C_ng, R_ng)
        numerator = sum(min(cand_counts[ng], ref_counts.get(ng, 0)) for ng in cand_counts)
        # Mẫu số: Tổng số n-gram trong candidate
        denominator = len(cand_ngrams) 
        
        if numerator == 0:
            return 0.0 # Đề bài: "If any precision is zero, BLEU is zero."
            
        precisions.append(numerator / denominator)
        
    # 2. Tính Brevity Penalty (BP)
    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)
        
    # 3. Kết hợp lại bằng trung bình nhân (Geometric mean)
    sum_log_p = sum(math.log(p) for p in precisions)
    geom_mean = math.exp((1 / max_n) * sum_log_p)
    
    bleu = bp * geom_mean
    
    return float(bleu)