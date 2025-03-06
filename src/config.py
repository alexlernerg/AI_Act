import os
from dotenv import load_dotenv

# Explicitly load config.env
load_dotenv("/content/AI_Act_Advisor/config.env")

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Print masked API keys for verification (optional)
print("OpenAI API Key:", OPENAI_API_KEY[:5] + "****")
print("Hugging Face API Key:", HF_API_KEY[:5] + "****")


# ✅ Root directory for the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Data paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "AI_Act.pdf")  # Modify if handling multiple PDFs

# ✅ Embeddings & FAISS index paths
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(EMBEDDINGS_DIR, "index.pkl")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "bge_embeddings.npy")

# ✅ Consistent Embedding Model for FAISS and Querying
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Ensure same model across all files
EMBEDDING_DIMENSION = 1024  # Set explicitly to avoid mismatches

# ✅ OpenAI Model Configuration
LLM_MODEL_NAME = "gpt-4-turbo"  # You can also use "gpt-4" or "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is securely stored

# ✅ FAISS Index Optimization
FAISS_HNSW_M = 32  # Controls graph connectivity in HNSW
FAISS_EF_CONSTRUCTION = 200  # Controls recall during index build
FAISS_EF_SEARCH = 50  # Higher value improves accuracy at runtime

# ✅ Hybrid Retrieval Weighting
FAISS_WEIGHT = 0.7  # FAISS contributes 70% to retrieval ranking
BM25_WEIGHT = 0.3  # BM25 contributes 30% to retrieval ranking
TOP_K = 10  # Number of results to retrieve

# ✅ Query Expansion Model (T5-based)
QUERY_EXPANSION_MODEL = "google/t5-small-ssm-nq"
QUERY_EXPANSION_TOP_K = 3  # Number of expanded queries

# ✅ Ensure paths exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ✅ Debugging Helper
def print_config():
    """Prints key config values for debugging."""
    print(f"✅ ROOT_DIR: {ROOT_DIR}")
    print(f"✅ DATA_DIR: {DATA_DIR}")
    print(f"✅ PDF_PATH: {PDF_PATH}")
    print(f"✅ EMBEDDINGS_DIR: {EMBEDDINGS_DIR}")
    print(f"✅ FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"✅ EMBEDDINGS_FILE: {EMBEDDINGS_FILE}")
    print(f"✅ EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME} (Dim: {EMBEDDING_DIMENSION})")
    print(f"✅ LLM_MODEL_NAME: {LLM_MODEL_NAME}")
    print(f"✅ FAISS_HNSW_M: {FAISS_HNSW_M}, FAISS_EF_CONSTRUCTION: {FAISS_EF_CONSTRUCTION}, FAISS_EF_SEARCH: {FAISS_EF_SEARCH}")
    print(f"✅ FAISS_WEIGHT: {FAISS_WEIGHT}, BM25_WEIGHT: {BM25_WEIGHT}, TOP_K: {TOP_K}")
    print(f"✅ QUERY_EXPANSION_MODEL: {QUERY_EXPANSION_MODEL}, QUERY_EXPANSION_TOP_K: {QUERY_EXPANSION_TOP_K}")

# ✅ Print config when script is run
if __name__ == "__main__":
    print_config()
