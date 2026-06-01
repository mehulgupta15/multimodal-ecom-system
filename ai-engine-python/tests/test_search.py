import os
import torch
import numpy as np
from src.model import CLIPEngine
from src.search_index import ProductSearchIndex

def run_semantic_search_test():
    print("\n==============================================")
    print("=== INITIALIZING SEMANTIC SEARCH ENGINE ===")
    print("==============================================\n")
    
    # 1. Paths to your saved FAISS files
    index_prefix = "./data/products_vector_index"
    
    # Detect processing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Boot up the CLIP Engine for text encoding
    print("Waking up CLIP text encoder network...")
    clip_engine = CLIPEngine()
    
    # 3. Instantiate and load the saved search index from disk
    print("Loading vector index matrices into memory...")
    search_index = ProductSearchIndex(dimension=512)
    try:
        search_index.load(index_prefix)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the index files at {index_prefix}.faiss")
        print("Please make sure you ran 'python -m src.index_catalog' successfully first.")
        return

    print("\n🎉 Search Engine ready! Ready for queries.")
    print("------------------------------------------------")
    
    # 4. Interactive Query Loop
    while True:
        # Prompt the user for terminal input
        query_text = input("\nEnter search concept (or type 'exit' to quit): ").strip()
        
        if query_text.lower() == 'exit':
            print("Shutting down search tester. Fantastic job today!")
            break
            
        if not query_text:
            continue
            
        print(f"Analyzing prompt: '{query_text}'...")
        
        # 5. Convert user text string into a 512-D tensor via CLIP
        with torch.no_grad():
            text_features = clip_engine.extract_text_features(query_text)            
            # Extract to raw NumPy row vector and reshape to 2D matrix shape (1, 512)
            query_vector = text_features.cpu().numpy().flatten().reshape(1, -1)
            
        # 6. Execute geometric dot-product lookups across your catalog vectors
        # Captured as a single object to prevent unpacking crashes
        top_matches = search_index.search(query_vector, top_k=3)        
        
        # 7. Print the top matching results dynamically
        print(f"\n✨ Top Results for '{query_text}':")
        print("-" * 60)
        
        # Safely loop through whatever format top_matches returns
        for rank, match in enumerate(top_matches, 1):
            # If your index returns zipped elements, this handles them smoothly
            if isinstance(match, (tuple, list)) and len(match) == 2:
                val1, val2 = match
                # Check if score comes first or label comes first
                if isinstance(val1, (float, np.float32, np.float64)):
                    print(f" Rank {rank}: Match Score = {float(val1):.4f} | Product: {val2}")
                else:
                    print(f" Rank {rank}: Match Score = {float(val2):.4f} | Product: {val1}")
            else:
                # Fallback to print the raw result if it's structured uniquely
                print(f" Rank {rank}: {match}")
        print("-" * 60)

if __name__ == "__main__":
    run_semantic_search_test()