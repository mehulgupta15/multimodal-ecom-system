import os
# Force underlying math libraries to stay efficient per worker loop
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import io
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd

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
        first_request = await text_request_queue.get()
        batch = [first_request]
        
        await asyncio.sleep(0.02)

        while text_request_queue.qsize() > 0 and len(batch) < 32:
            batch.append(text_request_queue.get_nowait())
            
        queries = [item["query"] for item in batch]
        futures = [item["future"] for item in batch]
        target_k = batch[0]["k"]
        
        try:
            model = ml_models["model"]
            processor = ml_models["processor"]
            device = ml_models["device"]
            index = ml_models["index"]
            product_lookup = ml_models["product_lookup"]
            
            start_inference = time.time()
            
            inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            
            with torch.no_grad():
                text_outputs = model.text_model(**inputs)
                features = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
                features = model.text_projection(features)

            if hasattr(features, "detach"):
                features = features.detach()

            features = features / features.norm(p=2, dim=-1, keepdim=True)
            queries_np = features.cpu().numpy().astype('float32')
            
            total_inference_time_ms = (time.time() - start_inference) * 1000
            per_query_time = round(total_inference_time_ms / len(batch), 2)
            
            for i, future in enumerate(futures):
                if future.done():
                    continue
                    
                single_query_vector = queries_np[i : i + 1]
                distances, indices = index.search(single_query_vector, target_k)
                
                results = []
                for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:
                        continue
                    
                    resolved = product_lookup.get(int(idx), {"product_id": f"prod_{idx}", "title": f"Product Index {idx}", "tags": {}})
                    if isinstance(resolved, dict):
                        results.append({
                            "rank": rank + 1,
                            "product_id": resolved.get("product_id"),
                            "title": resolved.get("title"),
                            "category": resolved.get("category"),
                            "tags": resolved.get("tags"),
                            "similarity_score": round(float(distance), 5)
                        })
                    else:
                        results.append({
                            "rank": rank + 1,
                            "product_id": resolved,
                            "similarity_score": round(float(distance), 5)
                        })
                
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
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            for _ in range(len(batch)):
                text_request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("====== [STARTUP] Initializing Search Engine Infrastructure ======")
    start_time = time.time()
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ml_models["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        ml_models["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ml_models["device"] = device
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        raise e

    # Force absolute paths relative to execution roots to fix path splitting issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    index_path = os.path.join(project_root, "data", "products_vector_index.faiss")
    csv_path = os.path.join(project_root, "data", "products_catalog.csv")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file missing at: {index_path}")
    
    print("Loading FAISS Index...")
    ml_models["index"] = faiss.read_index(index_path)

    # Dictionary Map with Strict Fallback and Rich Metadata (Tags)
    product_lookup = {}
    if os.path.exists(csv_path):
        print(f"Reading Database strings directly from Ledger: {csv_path}")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            id_val = str(row['product_id']).strip() if 'product_id' in df.columns and pd.notna(row['product_id']) else str(idx)
            title_val = str(row['title']).strip() if 'title' in df.columns and pd.notna(row['title']) else f"Product {id_val}"
            img_val = str(row['image_path']).strip() if 'image_path' in df.columns and pd.notna(row['image_path']) else ""
            cat_val = str(row['category']).strip() if 'category' in df.columns and pd.notna(row['category']) else ""
            pred_cat = str(row['predicted_category']).strip() if 'predicted_category' in df.columns and pd.notna(row['predicted_category']) else ""
            color_val = str(row['color']).strip() if 'color' in df.columns and pd.notna(row['color']) else ""
            mat_val = str(row['material']).strip() if 'material' in df.columns and pd.notna(row['material']) else ""
            style_val = str(row['style']).strip() if 'style' in df.columns and pd.notna(row['style']) else ""

            product_lookup[int(idx)] = {
                "product_id": id_val,
                "title": title_val,
                "image_path": img_val,
                "category": cat_val,
                "tags": {
                    "category": pred_cat,
                    "color": color_val,
                    "material": mat_val,
                    "style": style_val
                }
            }
        
        ml_models["product_lookup"] = product_lookup
    else:
        print(f"[WARNING] Catalog CSV ledger not found at {csv_path}.")
        ml_models["product_lookup"] = {}

    print("Launching Background GPU Worker...")
    asyncio.create_task(gpu_batch_processor())

    elapsed = time.time() - start_time
    print(f"====== [STARTUP] Engine ready. Loaded in {elapsed:.2f}s ======")
    
    yield
    print("====== [SHUTDOWN] Clearing engine allocations ======")
    ml_models.clear()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Autonomous E-Commerce Semantic Search Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.router.lifespan_context = lifespan


@app.get("/")
def read_root():
    return {"status": "online", "engine": "Multimodal CLIP + FAISS [CUDA Enabled]"}


@app.post("/search")
async def search_products(
    text_query: Optional[str] = Form(None, description="Text string query to search for products"),
    image_file: Optional[UploadFile] = File(None, description="Binary image file upload to search for products"),
    k: int = 5
):
    if "model" not in ml_models or "index" not in ml_models:
        raise HTTPException(status_code=503, detail="Search engine models are not fully initialized.")
        
    is_image_present = image_file is not None and image_file.filename != ""
    
    if not text_query and not is_image_present:
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: You must provide either a valid 'text_query' or an 'image_file'."
        )
    
    try:
        if is_image_present:
            model = ml_models["model"]
            processor = ml_models["processor"]
            device = ml_models["device"]
            index = ml_models["index"]
            product_lookup = ml_models["product_lookup"]

            start_inference = time.time()
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
                resolved = product_lookup.get(int(idx), {"product_id": f"prod_{idx}", "title": f"Product Index {idx}", "tags": {}})
                if isinstance(resolved, dict):
                    results.append({
                        "rank": rank + 1,
                        "product_id": resolved.get("product_id"),
                        "title": resolved.get("title"),
                        "category": resolved.get("category"),
                        "tags": resolved.get("tags"),
                        "similarity_score": round(float(distance), 5)
                    })
                else:
                    results.append({
                        "rank": rank + 1,
                        "product_id": resolved,
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
            
        else:
            current_loop = asyncio.get_running_loop()
            user_future = current_loop.create_future()
            await text_request_queue.put({"query": text_query, "k": k, "future": user_future})
            response_data = await user_future
            return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal Engine Execution Failed: {str(e)}")