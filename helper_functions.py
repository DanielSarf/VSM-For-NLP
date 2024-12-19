import os
import random
from typing import List, Tuple, Callable
import nltk
import re
import numpy as np
import tkinter as tk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import string
from similarity_measures import l2_norm_distance, cosine_distance, jaccard_similarity

nltk.download('stopwords')
stopwords_set = set(nltk.corpus.stopwords.words('english'))

def generate_random_text(length: int = 100) -> str:
    # Random printable characters from ASCII range 32-126
    return ''.join(random.choice(string.printable) for _ in range(length))

def load_document(document_path: str) -> str:
    try:
        with open(document_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading document {document_path}: {e}")
        return ""

def load_documents_from_folder(folder_path: str) -> Tuple[List[str], List[str]]:
    document_texts = []
    document_names = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder path {folder_path} does not exist.")
        return document_texts, document_names
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            document_texts.append(load_document(file_path))
            document_names.append(file_name)
    
    return document_texts, document_names

def tokenize_text(document_text: str, delimiters: str, make_lowercase = True, remove_stopwords: bool = True, outlier_detection: bool = False) -> List[str]:
    if make_lowercase:
        document_text = document_text.lower()
    
    pattern = f"[{re.escape(delimiters)}]"
    
    if outlier_detection:
        tokens = re.split(f"([{'|'.join(re.escape(d) for d in delimiters)}])", document_text)
    else:
        tokens = re.split(pattern, document_text)
    
    tokens = [token for token in tokens if (token.strip()) or 
              (outlier_detection and (token in delimiters))]
    
    if remove_stopwords:
        tokens = [token for token in tokens if token.lower() not in stopwords_set or
                  (outlier_detection and (token in delimiters))]

    return tokens

def create_corpus(all_document_texts: List[str], delimiters: str, mode: str) -> List[str]:
    corpus = set()
    
    for document_text in all_document_texts:
        corpus |= set(tokenize_text(document_text, delimiters, True, remove_stopwords = True, outlier_detection = mode == "Outlier Detection"))

    return corpus

def split_text_on_max_length(text: str, max_length: int, tokenizer) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    chunked_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return chunked_texts

def handle_query_mode(query_vector: np.ndarray, document_vectors: np.ndarray, similarity_measure: Callable, text_input: str, document_names: List[str]) -> None:
    distances_from_query = [similarity_measure(vector, query_vector) for vector in document_vectors]
    
    results_window = tk.Toplevel()
    results_window.title("Query Results")
    
    if similarity_measure == l2_norm_distance:
        sorted_results = sorted(zip(document_names, distances_from_query), key=lambda x: x[1])  # Lower distance first for L2
    else:
        sorted_results = sorted(zip(document_names, distances_from_query), key=lambda x: x[1], reverse=True)  # Higher similarity first for others
    
    tk.Label(results_window, text=f"Query Input: {text_input}", wraplength=400, justify="left").pack(pady=10)
    results_frame = tk.Frame(results_window)
    results_frame.pack(fill=tk.BOTH, expand=True)
    
    for document_name, distance in sorted_results:
        if similarity_measure == l2_norm_distance:
            tk.Label(results_frame, text=f"{document_name} - Similarity: {1 / (1 + distance):.2f}").pack(anchor="w", padx=10)  # L2 Norm
        elif similarity_measure == cosine_distance:
            tk.Label(results_frame, text=f"{document_name} - Similarity: {(1 + distance) / 2:.2f}").pack(anchor="w", padx=10)
        elif similarity_measure == jaccard_similarity:
            tk.Label(results_frame, text=f"{document_name} - Similarity: {distance:.2f}").pack(anchor="w", padx=10)

def handle_outlier_detection_mode(document_vectors: np.ndarray, document_names: List[str], similarity_measure: Callable) -> Tuple[List[str], List[float]]:
    scores = []

    for i, vector in enumerate(document_vectors):
        other_vectors = np.concatenate([document_vectors[:i], document_vectors[i+1:]], axis=0)

        if similarity_measure == l2_norm_distance:
            mean_distance = np.mean([similarity_measure(vector, other) for other in other_vectors])

            scores.append(mean_distance)
        else:
            mean_similarity = np.mean([similarity_measure(vector, other) for other in other_vectors])

            scores.append(1 - mean_similarity)
    
    # Normalization
    scores = np.array(scores)
    score_min = np.min(scores)
    score_max = np.max(scores)
    score_range = score_max - score_min

    if score_range == 0:
        scores = np.zeros_like(scores)
    else:
        scores = (scores - score_min) / score_range

    # For Cosine Similarity, lower value -> higher outlier score
    sorted_indices = np.argsort(scores)[::-1]  # Reverse the order to get most outlying documents first
    sorted_outliers = [document_names[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Display results in a new window, without using ranks
    results_window = tk.Toplevel()
    results_window.title("Outlier Detection Results")

    for outlier, score in zip(sorted_outliers, sorted_scores):
        tk.Label(results_window, text=f"{outlier} (Outlier Score: {score:.2f})").pack(anchor="w", padx=10)

def handle_cluster_mode(k_value: int, document_vectors: np.ndarray, document_names: str) -> None:
    kmeans = KMeans(n_clusters=k_value, random_state=42)
    cluster_labels = kmeans.fit_predict(document_vectors)
    clusters = [[] for _ in range(k_value)]
    
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(document_vectors)
    
    fig, ax = plt.subplots()
    cluster_colors = plt.cm.get_cmap('tab10', len(clusters))
    
    for cluster_idx, cluster_docs in enumerate(clusters):
        cluster_points = reduced_vectors[cluster_docs]
        
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx + 1}", 
                   color=cluster_colors(cluster_idx))
    
    for i, name in enumerate(document_names):
        ax.annotate(name, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
    
    ax.set_title("Cluster Visualization")
    ax.legend()
    plt.show()