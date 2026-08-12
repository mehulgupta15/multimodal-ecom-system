import os
import json
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# 1. Paths Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "src", "labels_config.json")
CSV_PATH = os.path.join(BASE_DIR, "data", "products_catalog.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "data_images")

# 2. PyTorch Fast Image Dataset for Batching
class TaggingDataset(Dataset):
    def __init__(self, image_list, image_dir, processor):
        self.image_list = image_list
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), img_name
        except Exception:
            # Return dummy zero tensor if corrupted image
            return torch.zeros((3, 224, 224)), img_name

def extract_label_features(model, processor, labels, device):
    """Pre-computes text embeddings for candidate labels once on GPU."""
    prompts = [f"a photo of a product with {label}" for label in labels]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        features = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
        features = model.text_projection(features)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features

def run_fast_gpu_batch_tagging(batch_size: int = 32):
    print("=== Starting High-Speed GPU Batch Auto-Tagging ===")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Acceleration Context: {device.upper()}")

    # Load Label Config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Convert to numpy arrays so batch indexing works (list[np_array] fails, np_array[np_array] works)
    categories = np.array(config["categories"])
    colors     = np.array(config["attributes"]["colors"])
    materials  = np.array(config["attributes"]["materials"])
    styles     = np.array(config["attributes"]["styles"])

    # Load Model onto GPU VRAM
    print("Loading CLIP Model into VRAM...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Pre-compute Label Vectors ONCE on GPU (Lightning Fast!)
    print("Pre-computing Label Vectors on GPU...")
    cat_feats = extract_label_features(model, processor, categories, device)
    col_feats = extract_label_features(model, processor, colors, device)
    mat_feats = extract_label_features(model, processor, materials, device)
    sty_feats = extract_label_features(model, processor, styles, device)

    # Read Catalog Ledger
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Catalog CSV missing at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    required_cols = ["predicted_category", "color", "material", "style"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object")

    # Find Images to Process
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(all_images)} total image files.")

    # Create PyTorch DataLoader for Parallel GPU Batching
    dataset = TaggingDataset(all_images, IMAGE_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results_map = {}
    print(f"Processing in Parallel Batches of {batch_size} on RTX 4050 GPU...")

    # GPU Matrix Batching Loop
    with torch.no_grad():
        for batch_idx, (pixel_values, img_names) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            
            # Extract Image Features Batch (Shape: [Batch, 512])
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            img_feats = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
            img_feats = model.visual_projection(img_feats)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

            # Single GPU Matrix Multiplication for all 32 items simultaneously!
            cat_preds = categories[torch.argmax(torch.matmul(img_feats, cat_feats.T), dim=-1).cpu().numpy()]
            col_preds = colors[torch.argmax(torch.matmul(img_feats, col_feats.T), dim=-1).cpu().numpy()]
            mat_preds = materials[torch.argmax(torch.matmul(img_feats, mat_feats.T), dim=-1).cpu().numpy()]
            sty_preds = styles[torch.argmax(torch.matmul(img_feats, sty_feats.T), dim=-1).cpu().numpy()]

            for i, name in enumerate(img_names):
                results_map[name] = (cat_preds[i], col_preds[i], mat_preds[i], sty_preds[i])

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Processed {(batch_idx + 1) * batch_size} / {len(all_images)} images...")

    # Assign all predicted tags to DataFrame in ONE fast operation
    print("\nUpdating CSV Ledger...")
    for idx, row in df.iterrows():
        img_name = os.path.basename(str(row["image_path"]))
        if img_name in results_map:
            cat, col, mat, sty = results_map[img_name]
            df.loc[idx, ["predicted_category", "color", "material", "style"]] = [cat, col, mat, sty]

    df.to_csv(CSV_PATH, index=False)
    print("\n[SUCCESS] GPU Batch Tagging Complete!")
    print(f"Tagged {len(results_map)} products in catalog CSV!")

if __name__ == "__main__":
    run_fast_gpu_batch_tagging(batch_size=32)
