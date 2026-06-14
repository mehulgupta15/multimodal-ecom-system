import os
# Force underlying math libraries to stay efficient per worker loop
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import io
import pickle
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Global state dictionary for memory allocations
ml_models = {}

# High-speed asynchronous queue to buffer text requests for the GPU
text_request_queue = asyncio.Queue()


async def gpu_batch_processor():
    """
    Continuous background worker loop. Accumulates separate incoming text queries
    and processes them as a single batched array on the GPU to maximize throughput.
    """
    while True:
        # 1. Block until at least one request enters the queue
        first_request = await text_request_queue.get()
        batch = [first_request]
        
        # 🔥 Give the concurrent flood exactly 20ms to stack up inside the queue buffer
        await asyncio.sleep(0.02)

        # 2. Snatch any other requests that arrived in this window (Up to batch size 32)
        while text_request_queue.qsize() > 0 and len(batch) < 32:
            batch.append(text_request_queue.get_nowait())
            
        # Extract individual text targets and their respective completion hooks (Futures)
        queries = [item["query"] for item in batch]
        futures = [item["future"] for item in batch]
        target_k = batch[0]["k"]  # Take the requested k-depth context
        
        try:
            model = ml_models["model"]
            processor = ml_models["processor"]
            device = ml_models["device"]
            index = ml_models["index"]
            product_ids = ml_models["product_ids"]
            
            start_inference = time.time()
            
            # Tokenize and execute the entire text batch on the GPU at once!
            inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            
            with torch.no_grad():
                text_outputs = model.text_model(**inputs)
                features = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
                features = model.text_projection(features)

            if hasattr(features, "detach"):
                features = features.detach()

            # Normalize the batch arrays using L2 Norm for vector metrics
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            queries_np = features.cpu().numpy().astype('float32')
            
            total_inference_time_ms = (time.time() - start_inference) * 1000
            per_query_time = round(total_inference_time_ms / len(batch), 2)
            
            # 3. Slice the matrix output and perform fast parallel lookups across FAISS index
            for i, future in enumerate(futures):
                if future.done():
                    continue
                    
                single_query_vector = queries_np[i : i + 1]
                distances, indices = index.search(single_query_vector, target_k)
                
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
                
                # Hand result directly back to the unique connection thread
                future.set_result({
                    "meta": {
                        "search_type": "text",
                        "batched_execution": True,
                        "batch_size": len(batch),
                        "execution_time_ms": per_query_time,
                        "results_count": len(results)
                    },
                    "results": results
                })
                
        except Exception as e:
            # Route processing failures gracefully back to individual open requests
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            # Clear loop task counters
            for _ in range(len(batch)):
                text_request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    Loads heavy models and indices into memory once using absolute paths.
    """
    print("====== [STARTUP] Initializing Search Engine Infrastructure ======")
    start_time = time.time()
    
    # 1. Load CLIP Model and Processor onto CUDA GPU Acceleration Layer
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

    # SPIN UP BACKGROUND PROCESSING TASK BEFORE YIELD
    print("Launching Background GPU Multi-Request Batching Loop Engine...")
    asyncio.create_task(gpu_batch_processor())

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
    return {"status": "online", "engine": "Multimodal CLIP + FAISS [CUDA Accelerated Enabled]"}


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
        
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        
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
    Unified Multimodal Search Endpoint. Routes image requests to standard pipeline,
    and forwards incoming concurrent text strings directly to our GPU batch processor.
    """
    if "model" not in ml_models or "index" not in ml_models:
        raise HTTPException(status_code=503, detail="Search engine models are not fully initialized.")
        
    is_image_present = image_file is not None and image_file.filename != ""
    
    if not text_query and not is_image_present:
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: You must provide either a valid 'text_query' or an 'image_file'."
        )
    
    try:
        # --- VISION BRANCH (Synchronous processing per image upload) ---
        if is_image_present:
            model = ml_models["model"]
            processor = ml_models["processor"]
            device = ml_models["device"]
            index = ml_models["index"]
            product_ids = ml_models["product_ids"]

            start_inference = time.time()
            print(f"Executing Image Search Pipeline for file: {image_file.filename}")
            image_bytes = await image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                vision_outputs = model.vision_model(**inputs)
                features = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
                features = model.visual_projection(features)

            if hasattr(features, "detach"):
                features = features.detach()

            if len(features.shape) == 1:
                features = features.unsqueeze(0)
                
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            query_np = features.cpu().numpy().astype('float32')
            
            distances, indices = index.search(query_np, k)
            inference_time_ms = (time.time() - start_inference) * 1000
            
            results = []
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                resolved_id = product_ids[idx] if product_ids is not None and idx < len(product_ids) else int(idx)
                results.append({
                    "rank": rank + 1,
                    "product_id": resolved_id,
                    "similarity_score": round(float(distance), 5)
                })
                    
            return {
                "meta": {
                    "search_type": "image",
                    "execution_time_ms": round(inference_time_ms, 2),
                    "results_count": len(results)
                },
                "results": results
            }
            
        # --- TEXT BRANCH (Forward directly to our new Asynchronous Batching System) ---
        else:
            current_loop = asyncio.get_running_loop()
            user_future = current_loop.create_future()
            
            # Package query attributes and drop into background processing buffer
            await text_request_queue.put({"query": text_query, "k": k, "future": user_future})
            
            # Wait asynchronously until the GPU worker finishes calculations for this batch slice
            response_data = await user_future
            return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal Engine Execution Failed: {str(e)}")