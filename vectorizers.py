from typing import List
from helper_functions import tokenize_text, split_text_on_max_length
from collections import Counter
from tf_idf_score import calculate_tf, calculate_idf
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np

def tf_idf_vectorization(all_document_texts: List[str], corpus: set, delimiters: str, mode: str) -> np.ndarray:
    vectors = []

    tokenized_documents = []
    for document_text in all_document_texts:
        tokenized_documents.append(tokenize_text(document_text, delimiters, outlier_detection = mode == "Outlier Detection"))

    idf_set = {}
    for term in corpus:
        idf_set[term] = calculate_idf(term, tokenized_documents)

    for tokenized_document in tokenized_documents:
        document_vector = []

        tf = calculate_tf(tokenized_document)

        for term in corpus:
            document_vector.append(tf.get(term, 0) * idf_set.get(term, 0))

        vectors.append(document_vector)

    return np.array(vectors)

def count_vectorization(all_document_texts: List[str], corpus: set, delimiters: str, mode: str) -> np.ndarray:
    vectors = []

    for document_text in all_document_texts:
        document_vector = []

        document_tokens = tokenize_text(document_text, delimiters, outlier_detection = mode == "Outlier Detection")

        document_word_counts = Counter(document_tokens)

        for term in corpus:
            document_vector.append(document_word_counts.get(term, 0))

        vectors.append(document_vector)

    return np.array(vectors)
    
def binary_count_vectorization(all_document_texts: List[str], corpus: set, delimiters: str, mode: str) -> np.ndarray:
    vectors = []

    for document_text in all_document_texts:
        document_vector = []

        document_tokens = tokenize_text(document_text, delimiters, outlier_detection = mode == "Outlier Detection")

        document_word_counts = set(document_tokens)

        for term in corpus:
            if term in document_word_counts:
                document_vector.append(1)
            else:
                document_vector.append(0)

        vectors.append(document_vector)
    
    return np.array(vectors)

def SBERT_vectorization(all_document_texts: List[str], corpus: set, delimiters: str, mode: str) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("paraphrase-MiniLM-L6-v2")
    
    grouped_vectors = []

    for document_text in all_document_texts:
        document_vectors = []

        split_texts = split_text_on_max_length(document_text, model.max_seq_length, tokenizer)

        grouped_vectors.append(model.encode(split_texts))

    return grouped_vectors