from typing import List
import math

def calculate_tf(document_text_tokens: str) -> dict[str, float]:
    tf = {}
    
    for token in document_text_tokens:
        if token in tf:
            tf[token] += 1
        else:
            tf[token] = 1
    
    for token in tf:
        tf[token] = tf[token] / len(document_text_tokens)
    
    return tf

def calculate_idf(term: str, tokenized_documents: List[str]) -> float:
    num_docs_with_term = 0

    for tokenized_document in tokenized_documents:
        if term in tokenized_document:
            num_docs_with_term += 1

    # Smoothing parameter to avoid divide by 0 error
    alpha = 1e-6

    return math.log((len(tokenized_documents) + alpha) / (num_docs_with_term + alpha))