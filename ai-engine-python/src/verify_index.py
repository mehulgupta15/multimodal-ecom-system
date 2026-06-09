import pickle
import faiss
import numpy as np

def inspect_database():
    print("=== Analyzing Loaded Vector Database Assets ===")
    
    # Load your current files
    try:
        index = faiss.read_index("data/products_vector_index.faiss")
        with open("data/products_vector_index_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading assets: {e}")
        return

    total_vectors = index.ntotal
    print(f"Total Vectors in Index: {total_vectors}")
    print(f"Total Entries in Metadata Ledger: {len(metadata)}")
    
    if total_vectors < 2:
        print("❌ Error: Not enough vectors to build a variation space.")
        return

    # Reconstruct the first two vectors from FAISS to check for duplicates
    try:
        v1 = index.reconstruct(0)
        v2 = index.reconstruct(1)
        
        # Calculate the direct cosine similarity between your first two items
        similarity = np.dot(v1, v2)
        print(f"Cosine match between item 0 and item 1: {similarity:.4f}")
        
        if similarity > 0.999:
            print("⚠️ WARNING: Your saved vectors are identical duplicates! Your generation pipeline is feeding frozen data layers.")
        else:
            print("✅ Vectors are distinct. The problem is a state tracking mismatch in api.py.")
            
    except Exception as e:
        print(f"Could not reconstruct vectors (Index might be optimized/compressed): {e}")

if __name__ == "__main__":
    inspect_database()