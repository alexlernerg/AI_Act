import pickle
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from config import DATA_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL_NAME

# Ensure the embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load embedding model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"trust_remote_code": True},
    cache_folder="/content/huggingface"
)
print("Embedding model loaded.")

# Load text chunks
def load_chunks(file_path):
    """Loads structured longer text chunks from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Generate embeddings in batches
def embed_chunks_in_batches(chunks, batch_size=16):
    """Generates embeddings in smaller batches to prevent memory overload."""
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="🔄 Generating embeddings"):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.append(np.array(batch_embeddings, dtype=np.float32))
    return np.vstack(embeddings)  # Stack all batches together

if __name__ == "__main__":
    chunks_file = os.path.join(DATA_DIR, "chunks_optimized.txt")
    
    print("Loading text chunks...")
    text_chunks = load_chunks(chunks_file)

    if not text_chunks:
        print("No text chunks found! Check chunking process.")
        exit()

    print(f"Loaded {len(text_chunks)} structured chunks.")

    print("Generating embeddings...")
    embeddings = embed_chunks_in_batches(text_chunks)

    # Debugging: Print first embedding for sanity check
    print("Sample Embedding (First Chunk):", embeddings[0][:10])

    # Save embeddings
    embeddings_file = os.path.join(EMBEDDINGS_DIR, "bge_embeddings.npy")
    np.save(embeddings_file, embeddings)
    print(f"Saved embeddings to {embeddings_file}")

    # Save text chunks for FAISS retrieval
    faiss_chunks_path = os.path.join(EMBEDDINGS_DIR, "index.pkl")
    with open(faiss_chunks_path, "wb") as f:
        pickle.dump(text_chunks, f)
    print(f"Saved {len(text_chunks)} structured text chunks to {faiss_chunks_path}")
