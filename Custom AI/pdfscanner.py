import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_chunks_from_pdf(file_path, chunk_size=190):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

def build_faiss_index(chunks, index_path="faiss_index"):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# print(model.encode("Test sentence"))

if __name__ == "__main__":
    all_chunks = []
    for filename in ["ayur1.pdf"]:
        all_chunks += extract_chunks_from_pdf(filename)
    build_faiss_index(all_chunks)