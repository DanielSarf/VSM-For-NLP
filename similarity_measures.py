import numpy as np

def jaccard_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    assert v1.size == v2.size
    
    # Convert the vectors to binary just in case binary vectorization was not used as a bug
    v1_binary = (v1 > 0).astype(int)
    v2_binary = (v2 > 0).astype(int)

    intersection = np.sum(v1_binary & v2_binary)
    union = np.sum(v1_binary | v2_binary)

    # To avoid divide by zero error
    if union == 0:
        return 0.0
    
    return intersection / union

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    assert v1.size == v2.size

    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # To avoid divide by zero error
    if magnitude_product == 0:
        return 0.0
    
    return np.dot(v1, v2) / magnitude_product

def l2_norm_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    assert v1.size == v2.size

    difference = v1 - v2

    return np.sqrt(np.dot(difference, difference))