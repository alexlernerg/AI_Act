import os
import faiss
import numpy as np
import pickle
from config import EMBEDDINGS_DIR, FAISS_INDEX_PATH

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load embeddings and text chunks
embeddings_file = os.path.join(EMBEDDINGS_DIR, "bge_embeddings.npy")
chunks_file = os.path.join(EMBEDDINGS_DIR, "index.pkl")

print("Loading stored embeddings...")
embeddings = np.load(embeddings_file)
embedding_dim = embeddings.shape[1]  # Ensure FAISS uses the correct dimension
print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embedding_dim}.")

# Load text chunks for reference
print("Loading stored text chunks...")
with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)

if len(chunks) != embeddings.shape[0]:
    raise ValueError(f"Mismatch: {len(chunks)} text chunks but {embeddings.shape[0]} embeddings found!")

# Check if FAISS index exists and validate dimension
if os.path.exists(FAISS_INDEX_PATH):
    print("Checking existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    if index.d != embedding_dim:
        print(f" FAISS index dimension mismatch: Found {index.d}, Expected {embedding_dim}. Rebuilding index...")
        os.remove(FAISS_INDEX_PATH)  # Remove old FAISS index
        index = None  # Reset index

# Create FAISS index if it doesn't exist or was removed
if not os.path.exists(FAISS_INDEX_PATH):
    print("Creating a new FAISS index...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32)  # HNSW for fast retrieval
    index.hnsw.efConstruction = 200  # High recall during index build
    index.hnsw.efSearch = 50  # High accuracy during retrieval
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS Index Created Successfully at {FAISS_INDEX_PATH}")

# Reload and Validate FAISS Index
print("Validating FAISS index...")
index_test = faiss.read_index(FAISS_INDEX_PATH)

if index_test.ntotal != index.ntotal:
    raise ValueError("FAISS index validation failed: Embeddings count mismatch!")

print("FAISS index validated successfully.")
