import sys
import os
import json
import glob
from datetime import datetime

# Define directories
PROJECT_PATH = "/content/AI_Act_Advisor"  # Update if necessary
RETRIEVAL_LOGS_DIR = "embeddings/retrieval_logs"
LLM_LOGS_DIR = "logs"
EVAL_RESULTS_DIR = "logs"

# Ensure eval logs directory exists
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Update PYTHONPATH for module recognition
sys.path.append(PROJECT_PATH)
sys.path.append(os.path.join(PROJECT_PATH, "src"))  # Ensure `src/` is recognized

print("✅ PYTHONPATH updated:", sys.path)

# Function to get the latest file matching a prefix
def get_latest_file(directory, prefix):
    """Retrieve the latest file in a directory matching a prefix."""
    files = glob.glob(os.path.join(directory, f"{prefix}_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getctime)  # Get latest by creation time

# Function to load JSON data
def load_json(file_path):
    """Load JSON from file."""
    if not file_path:
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Simple exact match evaluation function
def evaluate_retrieval(retrieval_data):
    """Simple evaluation of retrieval performance based on exact match."""
    relevant_docs = [d[0] for d in retrieval_data.get("post_rerank_results", [])]
    retrieved_docs = [d[0] for d in retrieval_data.get("faiss_results", []) + retrieval_data.get("bm25_results", [])]
    
    if not relevant_docs or not retrieved_docs:
        return {"Recall@5": 0.0, "MRR@5": 0.0}
    
    # Simple Recall@5 (check if any of the top 5 retrievals match the relevant docs)
    recall_at_k = sum(1 for doc in relevant_docs[:5] if doc in retrieved_docs) / 5.0
    # Simple MRR@5 (mean reciprocal rank)
    mrr = sum(1.0 / (retrieved_docs.index(doc) + 1) for doc in relevant_docs[:5] if doc in retrieved_docs) / 5.0
    
    return {"Recall@5": recall_at_k, "MRR@5": mrr}

# Function to evaluate LLM response using a simple match (presence of important keywords)
def evaluate_llm_response(llm_data, retrieval_data):
    """Evaluate LLM response based on simple exact match of important keywords."""
    
    response_text = llm_data.get("llm_response", "").strip()
    retrieved_texts = " ".join(retrieval_data.get("retrieved_texts", [])).strip()

    if not response_text or not retrieved_texts:
        print("⚠️ Warning: Missing LLM or retrieval text. Skipping evaluation.")
        return {"faithfulness": 0.0, "completeness": 0.0}

    # Check for exact keyword matches between the response and retrieved text
    keywords_response = set(response_text.split())  # Convert to set for quick lookup
    keywords_retrieved = set(retrieved_texts.split())
    
    # Faithfulness: Percentage of LLM response tokens found in the retrieved text
    faithfulness = len(keywords_response & keywords_retrieved) / len(keywords_response) if keywords_response else 0.0
    
    # Completeness: Percentage of retrieved text tokens found in the LLM response
    completeness = len(keywords_response & keywords_retrieved) / len(keywords_retrieved) if keywords_retrieved else 0.0

    return {"faithfulness": faithfulness, "completeness": completeness}

# Main function to evaluate retrieval and LLM performance
def main():
    # Get latest retrieval and LLM log files
    latest_retrieval_log = get_latest_file(RETRIEVAL_LOGS_DIR, "retrieval")
    latest_llm_log = get_latest_file(LLM_LOGS_DIR, "llm_response")
    
    # Load data from log files
    retrieval_data = load_json(latest_retrieval_log)
    llm_data = load_json(latest_llm_log)
    
    # If data is missing, print an error and return
    if not retrieval_data or not llm_data:
        print("❌ Missing retrieval or LLM data. Exiting.")
        return
    
    # Evaluate retrieval metrics
    retrieval_metrics = evaluate_retrieval(retrieval_data)
    
    # Evaluate LLM response metrics
    llm_metrics = evaluate_llm_response(llm_data, retrieval_data)
    
    # Prepare results
    eval_results = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"),
        "retrieval_metrics": retrieval_metrics,
        "llm_metrics": llm_metrics,
        "latest_retrieval_log": latest_retrieval_log,
        "latest_llm_log": latest_llm_log
    }
    
    # Save results to JSON file
    eval_log_path = os.path.join(EVAL_RESULTS_DIR, f"eval_results_{eval_results['timestamp']}.json")
    with open(eval_log_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    
    # Print and log results
    print(f"📊 Evaluation results saved: {eval_log_path}")
    print(json.dumps(eval_results, indent=4))

if __name__ == "__main__":
    main()
