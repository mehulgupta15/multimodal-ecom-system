import os
import pickle
import faiss
import numpy as np

class ProductSearchIndex:
    def __init__(self, dimension: int = 512):
        """
        Initializes the Vector Search Index.
        FAISS IndexFlatIP uses Inner Product (Dot Product). 
        When vectors are normalized, Inner Product equals Cosine Similarity.
        """
        self.dimension = dimension
        # 1. Initialize the FAISS index for 512-dimensional floating-point vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 2. Maintain an in-memory mapping list
        # Position 'i' in this list corresponds to ID 'i' in the FAISS index.
        self.id_to_metadata = []

    def add_vectors(self, vectors: np.ndarray, metadata_list: list):
        """
        Adds a batch of vectors and their corresponding image paths/metadata to the index.
        
        :param vectors: A numpy array of shape (N, 512) and type float32.
        :param metadata_list: A list of length N containing string identifiers (e.g., image paths).
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("The number of vectors must match the number of metadata entries.")
        
        # Ensure vectors are float32 (FAISS requires this)
        vectors = vectors.astype('float32')
        
        # Pre-normalize vectors to unit length (L2 norm = 1)
        # This makes Inner Product mathematically identical to Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero if an empty vector somehow slips through
        norms[norms == 0] = 1.0 
        normalized_vectors = vectors / norms
        
        # Inject into FAISS and append to our ledger
        self.index.add(normalized_vectors)
        self.id_to_metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """
        Searches the index for the most similar vectors to the query_vector.
        
        :param query_vector: A numpy array of shape (1, 512) or (512,)
        :param top_k: Number of closest items to return
        :return: A list of tuples containing (metadata, similarity_score)
        """
        # Reshape to 2D array (1, 512) if it's passed as a 1D array
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        query_vector = query_vector.astype('float32')
        
        # Normalize the query vector as well
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if norm[0, 0] == 0:
            norm[0, 0] = 1.0
        normalized_query = query_vector / norm
        
        # faiss.search returns:
        # similarities: matrix of float scores
        # indices: matrix of internal integer IDs
        similarities, indices = self.index.search(normalized_query, top_k)
        
        results = []
        # Parse the output matrices (we only sent 1 query vector, so look at index 0)
        for score, idx in zip(similarities[0], indices[0]):
            # FAISS returns -1 if it can't find enough items to fulfill top_k
            if idx != -1 and idx < len(self.id_to_metadata):
                metadata = self.id_to_metadata[idx]
                results.append((metadata, float(score)))
                
        return results

    def save(self, filepath_prefix: str):
        """
        Saves the FAISS index binary and the metadata tracking list to disk.
        """
        # Save the heavy optimization index matrix
        faiss.write_index(self.index, f"{filepath_prefix}.faiss")
        
        # Save the matching python metadata ledger
        with open(f"{filepath_prefix}_metadata.pkl", "wb") as f:
            pickle.dump(self.id_to_metadata, f)
        print(f"--- Index successfully saved to {filepath_prefix}.faiss/.pkl ---")

    def load(self, filepath_prefix: str):
        """
        Loads a saved FAISS index and its metadata list from disk.
        """
        if not os.path.exists(f"{filepath_prefix}.faiss"):
            raise FileNotFoundError(f"Could not find index file: {filepath_prefix}.faiss")
            
        self.index = faiss.read_index(f"{filepath_prefix}.faiss")
        
        with open(f"{filepath_prefix}_metadata.pkl", "rb") as f:
            self.id_to_metadata = pickle.load(f)
        print(f"--- Index successfully loaded from disk! ({self.index.ntotal} vectors total) ---")