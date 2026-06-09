import os
import time
import io
import pickle
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Global state dictionary for memory allocations
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

    # 2. Locate the absolute path to your data directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
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
    Legacy/Alternative GET endpoint for text-only search queries.
    """
    if "model" not in ml_models or "index" not in ml_models:
        raise HTTPException(status_code=503, detail="Search engine models are not fully initialized.")

    try:
        model = ml_models["model"]
        processor = ml_models["processor"]
        device = ml_models["device"]
        index = ml_models["index"]
        product_ids = ml_models["product_ids"]

        start_inference = time.time()
        
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            query_vector = outputs.cpu().numpy().astype("float32")

        k = 5
        distances, indices = index.search(query_vector, k)
        inference_time_ms = (time.time() - start_inference) * 1000

        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            resolved_id = product_ids[idx] if product_ids is not None else int(idx)
            results.append({
                "rank": rank + 1,
                "product_id": resolved_id,
                "similarity_score": float(distance)
            })

        return {
            "meta": {"query": query, "execution_time_ms": round(inference_time_ms, 2), "results_count": len(results)},
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Engine Exception: {str(e)}")


@app.post("/search")
async def search_products(
    text_query: Optional[str] = Form(None, description="Text string query to search for products"),
    image_file: Optional[UploadFile] = File(None, description="Binary image file upload to search for products"),
    k: int = 5
):
    """
    Unified Multimodal Search Endpoint. Accepts EITHER a text string form field 
    OR a binary image file upload, routing them to their respective CLIP towers.
    """
    if "model" not in ml_models or "index" not in ml_models:
        raise HTTPException(status_code=503, detail="Search engine models are not fully initialized.")
        
    # Robust Check: Ensure a file was actually uploaded and isn't just an empty form field
    is_image_present = image_file is not None and image_file.filename != ""
    
    if not text_query and not is_image_present:
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: You must provide either a valid 'text_query' or an 'image_file'."
        )
    
    try:
        model = ml_models["model"]
        processor = ml_models["processor"]
        device = ml_models["device"]
        index = ml_models["index"]
        product_ids = ml_models["product_ids"]

        start_inference = time.time()
        
        # --- VISION BRANCH ---
        if is_image_present:
            print(f"Executing Image Search Pipeline for file: {image_file.filename}")
            image_bytes = await image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Direct vision model tower execution
                vision_outputs = model.vision_model(**inputs)
                # Raw visual embeddings pooler extraction
                features = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
                # Project features to the shared multimodal embedding dimension
                features = model.visual_projection(features)
                
        # --- TEXT BRANCH ---
        else:
            print(f"Executing Text Search Pipeline for query: '{text_query}'")
            inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                # Direct text model tower execution
                text_outputs = model.text_model(**inputs)
                # Raw text embeddings pooler extraction
                features = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
                # Project features to the shared multimodal embedding dimension
                features = model.text_projection(features)

        # Global Safety Step: Ensure we have a raw, naked PyTorch Tensor
        if hasattr(features, "detach"):
            features = features.detach()

        # Ensure embedding shape is exactly 2D [1, 512] for FAISS
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
            
        # Normalize features vector using L2 norm (Required for True Cosine Similarity)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        query_np = features.cpu().numpy().astype('float32')
        
        # Execute Vector search against FAISS index
        distances, indices = index.search(query_np, k)
        inference_time_ms = (time.time() - start_inference) * 1000
        
        # Format and resolve results
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            
            if product_ids is not None and idx < len(product_ids):
                resolved_id = product_ids[idx]
            else:
                resolved_id = int(idx)
            
            results.append({
                "rank": rank + 1,
                "product_id": resolved_id,
                "similarity_score": round(float(distance), 5)
            })
                
        return {
            "meta": {
                "search_type": "image" if is_image_present else "text",
                "execution_time_ms": round(inference_time_ms, 2),
                "results_count": len(results)
            },
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal Engine Execution Failed: {str(e)}")