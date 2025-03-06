import os
import re
import fitz  # PyMuPDF
from config import PDF_PATH, DATA_DIR

# ✅ Ensure output directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts and cleans text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = []

    for page in doc:
        page_text = page.get_text("text")  # Extract raw text
        cleaned_text = clean_text(page_text)
        text.append(cleaned_text)

    return "\n".join(text)

# ✅ Function to clean extracted text
def clean_text(text):
    """Cleans text by fixing broken words and removing extra spaces."""
    text = re.sub(r"-\n", "", text)  # Fix words split by hyphens across lines
    text = re.sub(r"\n+", " ", text)  # Convert multiple newlines to spaces
    text = re.sub(r"\s{2,}", " ", text)  # Remove extra spaces
    return text.strip()

# ✅ Function to chunk text efficiently
def chunk_text(text, chunk_size=2000, overlap=200):
    """Splits text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start += chunk_size - overlap  # Maintain context with overlap

    return chunks

if __name__ == "__main__":
    print("🔄 Extracting text from PDF...")
    full_text = extract_text_from_pdf(PDF_PATH)

    if not full_text:
        print("❌ No text extracted. Check the PDF file.")
        exit()

    print(f"✅ Extracted {len(full_text)} characters from PDF.")

    print("🔄 Splitting text into chunks...")
    text_chunks = chunk_text(full_text)

    print(f"✅ Created {len(text_chunks)} text chunks.")

    # ✅ Save cleaned chunks to file
    chunks_file_path = os.path.join(DATA_DIR, "chunks_optimized.txt")
    with open(chunks_file_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_chunks))

    print(f"✅ Saved text chunks to {chunks_file_path}")
