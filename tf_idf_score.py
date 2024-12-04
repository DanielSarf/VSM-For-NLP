import re
from typing import List
import math

delimiter_pattern = r"[ \t\n.,!?;:\"'(){}\[\]<>@#%^&*|/~+=\\-]+"

def calculate_tf(document_text: str) -> dict[str, float]:
    words = [word for word in re.split(delimiter_pattern, document_text) if word.strip()]

    tf = {}
    
    for word in words:
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1
    
    for word in tf:
        tf[word] = tf[word] / len(words)
    
    return tf

def calculate_idf(term: str, documents: List[str]) -> float:
    num_docs_with_term = 0

    for document_text in documents:
        words = [word for word in re.split(delimiter_pattern, document_text) if word.strip()]

        if term in words:
            num_docs_with_term += 1

    # Adding alpha in denominator avoids divide by 0 error
    alpha = 1e-6

    return math.log((len(documents) + alpha) / (num_docs_with_term + alpha))

def calculate_tf_idf_from_text(term: str, document_text: str, documents: List[str]) -> float:
    tf = calculate_tf(document_text)
    idf = calculate_idf(term, documents)

    return tf.get(term, 0) * idf

def calculate_tf_idf_from_tf(term: str, tf: dict[str, float], documents: List[str]) -> float:
    idf = calculate_idf(term, documents)

    return tf.get(term, 0) * idf