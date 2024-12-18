from vectorizers import tf_idf_vectorization, count_vectorization, binary_count_vectorization, SBERT_vectorization
from similarity_measures import cosine_distance, l2_norm_distance, jaccard_similarity
from helper_functions import load_documents_from_folder, load_document, create_corpus, handle_query_mode, handle_outlier_detection_mode, handle_cluster_mode

def start_process():
    delimiters = delimiter_var.get()         # Get delimiters from input
    doc_path = doc_path_var.get()            # Get document path
    mode = mode_var.get()                    # Get selected mode
    source = source_var.get()                # Get source (Text or Document)
    source_path = source_path_var.get()      # Get source (Text or Document)
    k_value = k_var.get()                    # Get K value for clustering
    vectorization = vectorization_var.get()  # Get vectorization method
    similarity = similarity_var.get()        # Get similarity measure

    vectorization_methods = {
        "TF-IDF Vectorization": tf_idf_vectorization,
        "Count Vectorization": count_vectorization,
        "Binary Count Vectorization": binary_count_vectorization,
        "SBERT Vectorization": SBERT_vectorization,
    }

    similarity_measures = {
        "Cosine Distance": cosine_distance,
        "L2 Norm Distance": l2_norm_distance,
        "Jaccard Similarity": jaccard_similarity
    }

    document_texts, document_names = load_documents_from_folder(doc_path)
    text_input = [(text_box.get("1.0", "end-1c") if source == "Text" else load_document(source_path))] if mode == "Query Mode" else []
    
    corpus = create_corpus(document_texts + text_input, delimiters, mode == "Outlier Detection") if vectorization != "SBERT Vectorization" else None
    
    vectors = vectorization_methods[vectorization](document_texts + text_input, corpus, delimiters, mode)
    document_vectors = vectors[:-1] if mode == "Query Mode" else vectors
    query_vector = vectors[-1] if mode == "Query Mode" else []

    if mode == "Query Mode":
        handle_query_mode(query_vector, document_vectors, similarity_measures[similarity], text_input[0], document_names)

    elif mode == "Outlier Detection":
        handle_outlier_detection_mode(document_vectors, document_names, similarity_measures[similarity])
        
    elif mode == "Cluster Visualization":
        handle_cluster_mode(int(k_value), document_vectors, document_names)

### Tkinter Code

import tkinter as tk
from tkinter import filedialog

def browse_folder() -> None:
    folder_path = filedialog.askdirectory()
    if folder_path:
        doc_path_var.set(folder_path)

def browse_file() -> None:
    file_path = filedialog.askopenfilename()
    if file_path:
        source_path_var.set(file_path)

def update_mode_options() -> None:
    if mode_var.get() == "Query Mode":
        source_frame.pack(fill=tk.X, padx=10, pady=5, before=vectorization_label)
        k_frame.pack_forget()
    elif mode_var.get() == "Cluster Visualization":
        k_frame.pack(fill=tk.X, padx=10, pady=5, before=vectorization_label)
        source_frame.pack_forget()
    else:
        source_frame.pack_forget()
        k_frame.pack_forget()

    root.update_idletasks()
    root.geometry("")

def update_source_selection() -> None:
    if source_var.get() == "Text":
        text_box.pack(fill=tk.X, padx=0, pady=0)
        source_path_frame.pack_forget()
    elif source_var.get() == "Document":
        source_path_frame.pack(fill=tk.X, padx=0, pady=0)
        text_box.pack_forget()

def update_similarity_options() -> None:
    if vectorization_var.get() == "Binary Count Vectorization":
        jaccard_radio.config(state=tk.NORMAL)
    else:
        jaccard_radio.config(state=tk.DISABLED)

# Main window
root = tk.Tk()
root.title("VSM for NLP Tasks")
root.geometry("500x600")

# Variables
delimiter_var = tk.StringVar(value=r" \t\n.,!?;:\"'(){}\[\]<>@#%^&*|/~+=\\-")
doc_path_var = tk.StringVar()
mode_var = tk.StringVar(value="Query Mode")
source_var = tk.StringVar(value="Text")
source_path_var = tk.StringVar()
k_var = tk.StringVar()
vectorization_var = tk.StringVar(value="TF-IDF Vectorization")
similarity_var = tk.StringVar(value="Cosine Distance")

# Delimiters
tk.Label(root, text="Enter Delimiters:").pack(pady=5, anchor=tk.W, padx=10)
tk.Entry(root, textvariable=delimiter_var).pack(fill=tk.X, padx=10, pady=5)

# Path to documents
tk.Label(root, text="Path to Documents:").pack(pady=5, anchor=tk.W, padx=10)
path_frame = tk.Frame(root, padx=0, pady=0)
path_frame.pack(fill=tk.X, padx=4, pady=4)
tk.Entry(path_frame, textvariable=doc_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
tk.Button(path_frame, text="Browse", command=browse_folder).pack(side=tk.LEFT, padx=5)

# Mode selection
tk.Label(root, text="Mode:").pack(pady=5, anchor=tk.W, padx=10)
modes_frame = tk.Frame(root, padx=0, pady=0)
modes_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Radiobutton(modes_frame, text="Query Mode", variable=mode_var, value="Query Mode", command=update_mode_options).pack(side=tk.LEFT)
tk.Radiobutton(modes_frame, text="Outlier Detection", variable=mode_var, value="Outlier Detection", command=update_mode_options).pack(side=tk.LEFT)
tk.Radiobutton(modes_frame, text="Cluster Visualization", variable=mode_var, value="Cluster Visualization", command=update_mode_options).pack(side=tk.LEFT)

# Source frame (specific to Query Mode)
source_frame = tk.Frame(root, padx=0, pady=0)
tk.Label(source_frame, text="Source:").pack(pady=0, anchor=tk.W, padx=0)
source_radio_frame = tk.Frame(source_frame, padx=0, pady=0)
source_radio_frame.pack(fill=tk.X, padx=0, pady=0)
tk.Radiobutton(source_radio_frame, text="Text", variable=source_var, value="Text", command=update_source_selection).pack(side=tk.LEFT, padx=0)
tk.Radiobutton(source_radio_frame, text="Document", variable=source_var, value="Document", command=update_source_selection).pack(side=tk.LEFT, padx=5)
source_path_frame = tk.Frame(source_frame, padx=0, pady=0)
source_path_frame.pack(fill=tk.X, padx=0, pady=0)
tk.Entry(source_path_frame, textvariable=source_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
tk.Button(source_path_frame, text="Browse", command=browse_file).pack(side=tk.LEFT, padx=5)
text_box = tk.Text(source_frame, height=5, width=40)

# K value (specific to Cluster Visualization)
k_frame = tk.Frame(root, padx=0, pady=0)
tk.Label(k_frame, text="K Value:").pack(pady=5, anchor=tk.W, padx=0)
tk.Entry(k_frame, textvariable=k_var).pack(fill=tk.X, padx=0, pady=5)

# Vectorization methods
vectorization_label = tk.Label(root, text="Vectorization Method:")
vectorization_label.pack(pady=5, anchor=tk.W, padx=10)
tk.Radiobutton(root, text="TF-IDF Vectorization", variable=vectorization_var, value="TF-IDF Vectorization", command=update_similarity_options).pack(anchor=tk.W, padx=20)
tk.Radiobutton(root, text="SBERT Vectorization", variable=vectorization_var, value="SBERT Vectorization", command=update_similarity_options).pack(anchor=tk.W, padx=20)
tk.Radiobutton(root, text="Count Vectorization", variable=vectorization_var, value="Count Vectorization", command=update_similarity_options).pack(anchor=tk.W, padx=20)
tk.Radiobutton(root, text="Binary Count Vectorization", variable=vectorization_var, value="Binary Count Vectorization", command=update_similarity_options).pack(anchor=tk.W, padx=20)

# Similarity measure
tk.Label(root, text="Similarity Measure:").pack(pady=5, anchor=tk.W, padx=10)
tk.Radiobutton(root, text="Cosine Distance", variable=similarity_var, value="Cosine Distance").pack(anchor=tk.W, padx=20)
tk.Radiobutton(root, text="L2 Norm Distance", variable=similarity_var, value="L2 Norm Distance").pack(anchor=tk.W, padx=20)
jaccard_radio = tk.Radiobutton(root, text="Jaccard Similarity", variable=similarity_var, value="Jaccard Similarity", state=tk.DISABLED)
jaccard_radio.pack(anchor=tk.W, padx=20)

# Start button
tk.Button(root, text="Start", command=start_process).pack(pady=20)

# Initial state
update_mode_options()
update_similarity_options()
update_source_selection()

# Run application
root.mainloop()