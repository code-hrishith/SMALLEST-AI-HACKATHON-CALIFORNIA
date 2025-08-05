from sentence_transformers import SentenceTransformer
import faiss
import pickle
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(query, index_path="/Users/hrishithsavir/Desktop/CustomAI/faiss_index", top_k=1):
    index = faiss.read_index(f"{index_path}/faiss.index")
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)

    results = [chunks[i] for i in I[0]]


    # return "\n\n".join(results) for multiple answers 

    return results[0]


