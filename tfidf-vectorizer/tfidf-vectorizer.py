import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    tokenized_docs = [doc.split() for doc in documents]
    unique_terms = set()
    for doc in tokenized_docs:
        unique_terms.update(doc)
    
    vocabulary = sorted(list(unique_terms))
    vocab_size = len(vocabulary)
    num_docs = len(documents)
    
    tf_matrix = np.zeros((num_docs, vocab_size))
    df_counts = np.zeros(vocab_size)
    
    for i, doc_tokens in enumerate(tokenized_docs):
        term_counts = Counter(doc_tokens)
        total_terms = len(doc_tokens)
        
        for j, term in enumerate(vocabulary):
            if term in term_counts:
                tf_matrix[i, j] = term_counts[term] / total_terms
                df_counts[j] += 1
                
    idf_vector = np.log(num_docs / df_counts)
    tfidf_matrix = tf_matrix * idf_vector
    
    return tfidf_matrix, vocabulary