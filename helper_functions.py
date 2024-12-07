import os
import random
from typing import List, Tuple
import nltk
import re

nltk.download('stopwords')
stopwords_set = set(nltk.corpus.stopwords.words('english'))

def generate_random_text(length: int = 100) -> str:
    # Random characters from byte values (0 to 255)
    return ''.join(chr(random.randint(0, 255)) for _ in range(length))

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

def tokenize_text(document_text: str, delimiters: str, make_lowercase = True, remove_stopwords: bool = True, outlier_detection: bool = False) -> list:
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
        tokens = [token for token in tokens if tokens.lower() not in stopwords_set or
                  (outlier_detection and (token in delimiters))]

    return tokens

def create_corpus(all_document_texts: List[str], delimiters: str, mode: str) -> List[str]:
    corpus = set()
    
    for document_text in all_document_texts:
        corpus += tokenize_text(document_text, delimiters, True, remove_stopwords = True, outlier_detection = mode == "Outlier Detection")

    return corpus

def split_text_on_max_length(text: str, max_length: int, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    chunked_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return chunked_texts