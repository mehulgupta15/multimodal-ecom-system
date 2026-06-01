import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Core custom engine components
from src.model import CLIPEngine
from src.dataset import ECommerceDataset
from src.search_index import ProductSearchIndex

def get_transforms():
    """
    Standard normalization transforms for CLIP input images.
    Resizes image to 224x224 and applies CLIP specific color-channel normalization.
    """
    return Compose([
        Resize(224, interpolation=3),  # Bicubic interpolation
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def run_production_indexing():
    print("=== Starting Production Catalog Vector Extraction Pipeline ===")
    
    # 1. Physical file paths on your E: drive
    csv_path = "./data/products_catalog.csv"
    img_dir = "./data/data_images"
    index_output_prefix = "./data/products_vector_index"
    
    # Ensure database output directory exists
    os.makedirs(os.path.dirname(index_output_prefix), exist_ok=True)
    
    # 2. Critical Validation Check
    if not os.path.exists(csv_path):
        print(f"❌ Production Error: Could not find catalog ledger at '{csv_path}'.")
        print("Please run 'python -m src.scraper' first to download assets!")
        return

    # Verify that we actually have a non-empty catalog spreadsheet
    try:
        df_check = pd.read_csv(csv_path)
        if df_check.empty:
            print("❌ Production Error: The catalog CSV file is completely empty.")
            return
        print(f"Verified spreadsheet database: Found {len(df_check)} registered items.")
    except Exception as e:
        print(f"❌ Failed reading catalog ledger layout: {e}")
        return

    # 3. Compute Resource Evaluation (CPU vs GPU activation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Compute Context: {device.upper()}")
    
    # 4. Neural Network Engine and Database Initialization
    print("Loading CLIP Neural Network Engine...")
    clip_engine = CLIPEngine()
    
    print("Initializing FAISS Vector Index (512-Dimensions)...")
    search_index = ProductSearchIndex(dimension=512)
    
    # 5. Build PyTorch Pipeline over Real Image Directory
    print("Assembling PyTorch data dataset and transform streams...")
    transform = get_transforms()
    
    try:
        dataset = ECommerceDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
    except Exception as e:
        print(f"❌ Pipeline Assembly Failure: Dataset instantiation crashed. Details: {e}")
        return
        
    # Batch size of 4 optimizes matrix throughput over local disk boundaries safely
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_vectors = []
    all_metadata = []
    
    print(f"Beginning deep feature extraction loop across items...")
    
    # 6. Tensor Computation Loop (No-grad context eliminates memory leaks)
    with torch.no_grad():
        for batch_idx, (batch_images, batch_paths) in enumerate(dataloader):
            try:
                # Dispatch tensor stack to target hardware device context
                batch_images = batch_images.to(device)
                
                # Extract the 512-dimensional normalized vectors from raw image pixels
                # NOTE: If your model class calls it 'get_image_features', keep this line. 
                # If your class uses 'extract_image_features', change it below.
                image_features = clip_engine.extract_image_features(batch_images)
                
                # Collect compute variables from VRAM back into host system RAM memory
                vectors = image_features.cpu().numpy()
                
                all_vectors.append(vectors)
                all_metadata.extend(batch_paths)
                
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
            except Exception as e:
                print(f"⚠️ Warning: Batch execution anomaly encountered at segment {batch_idx + 1}: {e}")
                continue

    if len(all_vectors) == 0:
        print("❌ Critical Failure: Zero vectors generated during data sweep. Aborting write.")
        return

    # 7. Stack arrays vertically to compile the master matrix ledger
    final_vectors = np.vstack(all_vectors).astype('float32')
    
    print("Registering features into FAISS index matrix space...")
    search_index.add_vectors(final_vectors, all_metadata)
    
    # 8. Commit structural files permanently to the local disk layout
    search_index.save(index_output_prefix)
    print("\n🎉 === Production Database Built & Locked! ===")
    print(f"Registered Total Elements: {len(all_metadata)}")
    print(f"Target Destination Files: {index_output_prefix}.faiss / .pkl")

if __name__ == "__main__":
    run_production_indexing()