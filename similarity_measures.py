import numpy as np
from typing import Union, Literal
from helper_functions import document_vectorizer

# def jaccard_distance(d1, d2) -> float:
#     pass

def cosine_distance(d1: Union[str, np.ndarray], d2: Union[str, np.ndarray], docs_length_difference_handling: Literal["truncate", "padding"] = "truncate") -> float:
    d1, d2 = [d if isinstance(d, np.ndarray) else document_vectorizer(d) for d in [d1, d2]]

    if d1.size != d2.size:
        pass #Todo

    magnitude_product = np.linalg.norm(d1) * np.linalg.norm(d2)
    
    # To avoid divide by zero error
    if magnitude_product == 0:
        return 0.0
    
    return np.dot(d1, d2) / magnitude_product

def l2_norm_distance(d1: Union[str, np.ndarray], d2: Union[str, np.ndarray], docs_length_difference_handling: Literal["truncate", "padding"] = "truncate") -> float:
    d1, d2 = [d if isinstance(d, np.ndarray) else document_vectorizer(d) for d in [d1, d2]]
    
    if d1.size != d2.size:
        pass #Todo

    difference = d1 - d2

    return np.sqrt(np.dot(difference, difference))