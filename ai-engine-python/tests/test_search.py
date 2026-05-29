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
    device = "cuda" if torch.torch.cuda.is_available() else "cpu"
    
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
            # tokenize_text and get_text_features are part of your Day 4 engine mechanics
            text_features = clip_engine.extract_text_features(query_text)            
            # Extract to raw NumPy row vector
            query_vector = text_features.cpu().numpy().flatten()
            
        # 6. Execute geometric dot-product lookups across your catalog vectors
        top_matches = search_index.search(query_vector, top_k=3)
        
        # 7. Print the top matching filenames alongside their similarity scores
        print(f"\n✨ Top 3 Semantic Results for '{query_text}':")
        print("-" * 60)
        for rank, (img_path, score) in enumerate(top_matches, 1):
            # Similarity score represents cosine similarity (closer to 1.0 = better match)
            print(f" Rank {rank}: Match Score = {score:.4f} | Item: {img_path}")
        print("-" * 60)

if __name__ == "__main__":
    run_semantic_search_test()