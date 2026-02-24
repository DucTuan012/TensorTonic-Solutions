import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([])
        
    doc_lengths = np.array([len(doc) for doc in docs])
    avgdl = np.mean(doc_lengths)
    
    if avgdl == 0:
        avgdl = 1e-9 
        
    scores = np.zeros(N)
    
    for term in set(query_tokens):
        tf_array = np.array([doc.count(term) for doc in docs])
        df = np.count_nonzero(tf_array)
        
        if df == 0:
            continue
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf_array * (k1 + 1)
        denominator = tf_array + k1 * (1 - b + b * (doc_lengths / avgdl))
        scores += idf * (numerator / denominator)
        
    return scores