import os
import faiss
import pickle
import numpy as np
import openai
import torch
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from datetime import datetime
from config import FAISS_INDEX_PATH, EMBEDDINGS_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

RETRIEVAL_RESULTS_PATH = os.path.join(EMBEDDINGS_DIR, "retrieval_results.json")
RETRIEVAL_LOGS_DIR = os.path.join(EMBEDDINGS_DIR, "retrieval_logs")
os.makedirs(RETRIEVAL_LOGS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Load Environment Variables from `config.env`
load_dotenv("config.env")  # Explicitly specify the file

# Ensure API Key is Set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY. Set it as an environment variable.")

# Load FAISS Index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"❌ FAISS index not found at {FAISS_INDEX_PATH}. Run indexing first.")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
print("FAISS index loaded.")

#Load stored text chunks
chunks_file = os.path.join(EMBEDDINGS_DIR, "index.pkl")
print("Loading stored text chunks...")
with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)
print(f"Loaded {len(chunks)} text chunks.")

# Prepare BM25 Corpus
bm25_corpus = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(bm25_corpus)

# Load embedding model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# Load Faster Re-Ranking Model
FAST_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(FAST_RERANKER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Re-ranking model loaded: {FAST_RERANKER_MODEL}")

# Optimized Hybrid Retrieval Function
# Load an additional embedding model for query similarity scoring
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def score_retrieved_chunks(query, retrieved_texts):
    """Ranks retrieved texts based on semantic similarity to the query."""
    query_embedding = similarity_model.encode([query], normalize_embeddings=True)
    text_embeddings = similarity_model.encode(retrieved_texts, normalize_embeddings=True)

    # Compute cosine similarity between query and retrieved chunks
    similarity_scores = np.dot(text_embeddings, query_embedding.T).flatten()

    # Sort texts by highest similarity score
    sorted_texts = [x for _, x in sorted(zip(similarity_scores, retrieved_texts), reverse=True)]
    return sorted_texts[:10]  # Keep only top 10 most relevant chunks

def get_hybrid_retrieval(query, top_k=10, faiss_weight=0.7, bm25_weight=0.3, rerank_k=5):
    """Generalized retrieval pipeline: FAISS + BM25 + Transformer Re-Ranking (Query-Aware)"""
    
    print("Searching FAISS index...")
    query_embedding = np.array(embedding_model.embed_query(query), dtype=np.float32).reshape(1, -1)
    faiss_scores, faiss_indices = index.search(query_embedding, top_k)

    faiss_results = {}
    if faiss_scores is not None and len(faiss_scores) > 0 and len(faiss_indices[0]) > 0:
        faiss_scores = faiss_scores[0]
        if np.max(faiss_scores) > np.min(faiss_scores):  # Avoid division by zero
            faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores) + 1e-9)
        faiss_results = {chunks[idx]: faiss_scores[i] * faiss_weight for i, idx in enumerate(faiss_indices[0]) if 0 <= idx < len(chunks)}

    print("Searching BM25 index...")
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    bm25_results = {}
    if np.max(bm25_scores) > np.min(bm25_scores):  # Avoid division by zero
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        bm25_results = {chunks[idx]: bm25_scores[idx] * bm25_weight for idx in bm25_indices}

    # Combine FAISS & BM25 Results
    combined_results = {**faiss_results, **bm25_results}

    # Apply Query-Aware Re-Ranking Before Transformer Re-Ranking (Ensure function exists)
    if "score_retrieved_chunks" in globals():
        filtered_results = score_retrieved_chunks(query, list(combined_results.keys()))
    else:
        filtered_results = list(combined_results.keys())

    # Limit to `rerank_k` before re-ranking
    top_rerank_candidates = sorted(filtered_results, key=lambda x: combined_results.get(x, 0), reverse=True)[:rerank_k]

    # Fast Batch Re-Ranking
    print(f"Re-ranking {rerank_k} candidates...")
    rerank_inputs = [(query, chunk) for chunk in top_rerank_candidates]
    rerank_scores = reranker.predict(rerank_inputs, batch_size=8)

    # Sort by Re-Rank Score
    ranked_results = [x for _, x in sorted(zip(rerank_scores, top_rerank_candidates), reverse=True)]

    print("Re-ranking completed (Optimized).")

    # Store retrieval outputs with a timestamped filename
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"retrieval_{timestamp}.json"
    file_path = os.path.join(RETRIEVAL_LOGS_DIR, file_name)

    retrieval_data = {
        "query": query,
        "timestamp": timestamp,
        "faiss_results": sorted(faiss_results.items(), key=lambda x: x[1], reverse=True)[:top_k],
        "bm25_results": sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)[:top_k],
        "pre_rerank_candidates": top_rerank_candidates,
        "post_rerank_results": ranked_results[:top_k]
    }

    # Save each query result as a separate file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_data, f, indent=4, ensure_ascii=False)

    print(f"Retrieval result saved to: {file_path}")

    return ranked_results[:top_k]



# Load Summarization Model (BART-Large-CNN)
print("Loading BART-Large-CNN summarization model...")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
print("Summarization model loaded.")

# Chatbot Memory
chat_memory = []

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists

def llm_generate_answer(query, retrieved_texts):
    """Generates an answer using LLM while considering full retrieved context."""

    if not retrieved_texts:
        return "**No relevant information found in the retrieved legal text.**"

    chat_history = "\n".join(chat_memory[-5:])  # Keep last 5 exchanges
    context = "\n\n".join(retrieved_texts[:5])  # Expanding the number of retrieved texts

    # Summarize retrieved text to highlight key legal points
    summary_prompt = f"""
    Summarize the following retrieved legal text from the AI Act.
    Focus on key provisions, requirements, and any relevant clarifications. Include the section in the text if referenced.

    **Retrieved Legal Text:**  
    {context}

    **Summarized Key Points:**
    """

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        summary_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0
        )

        summarized_context = summary_response.choices[0].message.content

    except Exception as e:
        summarized_context = "**Error summarizing retrieved text. Using raw text instead.**"
        print(f"⚠️ Summarization Error: {str(e)}")

    # Generalized prompt that works for any legal question
    prompt = f"""
    You are a strict legal AI assistant specializing in the European AI Act.  
    Your task is to provide **precise, regulation-backed answers** while ensuring all responses **contain fine amounts and penalties if applicable**.  

    ## **User Question:** {query}  

    ## **Guidelines for Your Answer:**
    1. **Fines & Penalties Must Be Explicit**  
      - If the retrieved legal text includes fines or sanctions, **always provide exact amounts (EUR) and applicable percentages.**  
      - If penalties vary by company size (e.g., SMEs vs large enterprises), **explain how they differ**.  

    2. **Legal References Are Mandatory**  
      - Always **cite specific articles** from the AI Act (e.g., Article 5, Article 49(2)).  
      - If information is in **Annexes (e.g., Annex III for high-risk AI)**, mention where to check.  

    3. **Handling Missing Information**  
      - If the **retrieved legal text does not include fines**, **run a secondary check** specifically for penalties.  
      - If still unavailable, **explicitly state that the legal text does not specify penalties for this case.**  
      - Never speculate—only provide regulation-backed answers.  

    4. **Clear Structure**  
      - Always present answers in a structured format:  
        - **1️⃣ Risk Classification**  
        - **2️⃣ Obligations**  
        - **3️⃣ Fines & Penalties**  
        - **4️⃣ Enforcement & Compliance Audits**  

    5. **Examples & Precedents** (If Available)  
      - If past EU AI regulation enforcement cases exist, **mention them** to support your answer.  


    **Summary of relevant legal text, include all key figures and references to sections the text:**
    {summarized_context}

    **Now, generate a well-structured response based on the retrieved text.**
    - Use bullet points for clarity.
    - Highlight key legal provisions or requirements.
    - If relevant, explain any conditions or exceptions mentioned in the text.
    - If the retrieved text lacks necessary details, state what is missing.
    - Avoid speculating; provide only the information present in the retrieved text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are a strict legal AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )

        response_text = response.choices[0].message.content
        chat_memory.append(f"User: {query}\nAssistant: {response_text}")  # Store conversation

        # Log the full retrieval and LLM output for evaluations
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(LOGS_DIR, f"llm_response_{timestamp}.json")

        log_data = {
            "timestamp": timestamp,
            "query": query,
            "retrieved_texts": retrieved_texts[:5],
            "summarized_context": summarized_context,
            "llm_response": response_text
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(f"📁 LLM Response logged at: {log_file}")
        return response_text
    except Exception as e:
        return f"❌ LLM API Error: {str(e)}"



# Chatbot Function
def chat():
    """Runs an interactive chatbot loop with memory."""
    print("\n🤖 **AI Act Legal Chatbot (Optimized)** 🤖")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        retrieved_texts = get_hybrid_retrieval(query)

        print("\n**Retrieved Texts:**\n")
        for i, text in enumerate(retrieved_texts, 1):
            print(f"{i}. {text[:300]}...\n")  # Truncate for readability

        print(f"Retrieval results saved to {RETRIEVAL_RESULTS_PATH}")
        structured_answer = llm_generate_answer(query, retrieved_texts)

        print("\n**AI:**\n")
        print(structured_answer)

# ✅ Run Chatbot
if __name__ == "__main__":
    chat()
