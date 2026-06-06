import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle  # Or json, depending on how you stored your ID mappings

# Global state dictionary to clean up references if needed
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    Loads heavy models and indices into memory once using absolute paths.
    """
    print("====== [STARTUP] Initializing Search Engine Infrastructure ======")
    start_time = time.time()
    
    # 1. Load CLIP Model and Processor onto CPU or GPU
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model onto device: {device}...")
        ml_models["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        ml_models["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ml_models["device"] = device
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        raise e

    # 2. Dynamically locate the absolute path to your data directory
    # Grabs the folder where api.py lives (src/) and steps up one level to project root
    # 2. Locate the absolute path to your data directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # MATCHING YOUR EXACT VISUAL FILE SYSTEM NAMES:
    index_path = os.path.join(project_root, "data", "products_vector_index.faiss")
    metadata_path = os.path.join(project_root, "data", "products_vector_index_metadata.pkl")
    
    print(f"Targeting FAISS index at absolute path: {index_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file missing at: {index_path}")
    
    print("Loading FAISS index into memory...")
    ml_models["index"] = faiss.read_index(index_path)

    # 3. Load product lookup metadata
    if os.path.exists(metadata_path):
        print(f"Loading product lookup metadata from: {metadata_path}")
        with open(metadata_path, "rb") as f:
            ml_models["product_ids"] = pickle.load(f)
    else:
        print(f"[WARNING] Metadata file missing at {metadata_path}. Metrics will use fallback indices.")
        ml_models["product_ids"] = None

    elapsed = time.time() - start_time
    print(f"====== [STARTUP] Engine ready. Infrastructure loaded in {elapsed:.2f}s ======")
    
    yield
    
    # Clean up on shutdown
    print("====== [SHUTDOWN] Clearing engine allocations ======")
    ml_models.clear()


# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="Autonomous E-Commerce Semantic Search Engine",
    version="1.0.0"
)

app.router.lifespan_context = lifespan


@app.get("/")
def read_root():
    return {"status": "online", "engine": "Multimodal CLIP + FAISS"}


@app.get("/search")
async def search(query: str = Query(..., min_length=1, description="Text query to search for products")):
    """
    Takes an HTTP string query, generates a text vector via CLIP, 
    and searches the pre-loaded FAISS index for nearest neighbors.
    """
    if "model" not in ml_models or "index" not in ml_models:
        raise HTTPException(status_code=503, detail="Search engine models are not fully initialized.")

    try:
        model = ml_models["model"]
        processor = ml_models["processor"]
        device = ml_models["device"]
        index = ml_models["index"]
        product_ids = ml_models["product_ids"]

        # 1. Benchmark vector generation step
        start_inference = time.time()
        
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 1. Compute CLIP features
            outputs = model.get_text_features(**inputs)
            
            # 2. Extract raw PyTorch tensor regardless of the wrapper layer type
            if hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            elif isinstance(outputs, dict) and "text_embeds" in outputs:
                text_features = outputs["text_embeds"]
            elif hasattr(outputs, "last_hidden_state"):
                # Fallback if text features pooling isn't explicitly detached
                text_features = outputs.last_hidden_state[:, 0, :]
            else:
                text_features = outputs

            # 3. Perform Vector Math & Data Type Normalization
            # Ensure it is a 2D tensor for FAISS compatibility [1, 512]
            if len(text_features.shape) == 1:
                text_features = text_features.unsqueeze(0)
                
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().astype("float32")

        # 2. Query FAISS index (Retrieve top 5 closest items)
        k = 5
        distances, indices = index.search(query_vector, k)
        inference_time_ms = (time.time() - start_inference) * 1000

        # 3. Construct structured JSON response payload
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue  # FAISS returns -1 if fewer items exist than requested k
                
            # Resolve actual database ID if tracking dictionary is present
            resolved_id = product_ids[idx] if product_ids is not None else int(idx)
            
            results.append({
                "rank": rank + 1,
                "product_id": resolved_id,
                "similarity_score": float(distance)  # cast from numpy float to native float
            })

        return {
            "meta": {
                "query": query,
                "execution_time_ms": round(inference_time_ms, 2),
                "results_count": len(results)
            },
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Engine Exception: {str(e)}")