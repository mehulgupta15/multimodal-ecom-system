import os
import json
# pyrefly: ignore [missing-import]
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "src", "labels_config.json")
CSV_PATH = os.path.join(BASE_DIR, "data", "products_catalog.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "data_images")

# 1. Load the Configuration Blueprint
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

categories = config["categories"]
colors = config["attributes"]["colors"]
materials = config["attributes"]["materials"]
styles = config["attributes"]["styles"]

# 2. Automatically detect your RTX 4050 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running AI pipeline on device: {device}")

# 3. Initialize CLIP and send weights to GPU VRAM
print("Loading CLIP model components...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def get_best_match(image, candidate_labels):
    prompts = [f"a photo of a product with {label}" for label in candidate_labels]
    
    # Generate tensors and push them straight to the GPU
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    return candidate_labels[probs.argmax()]

def process_batch_tagging():
    if not os.path.exists(IMAGE_DIR) or not os.listdir(IMAGE_DIR):
        print(f"No images found in {IMAGE_DIR}. Run the scraper first!")
        return

    # Load your existing structural CSV
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            df = pd.DataFrame(columns=["product_id", "title", "image_path", "predicted_category", "color", "material", "style"])
    else:
        df = pd.DataFrame(columns=["product_id", "title", "image_path", "predicted_category", "color", "material", "style"])

    # Ensure all baseline columns exist safely and are object dtype for string assignment
    required_columns = ["product_id", "title", "image_path", "predicted_category", "color", "material", "style"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object")

    images_to_process = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(images_to_process)} images. Starting batch processing...")

    for img_name in images_to_process:
        # Match using basename of 'image_path' column
        matches = df[df["image_path"].apply(lambda x: os.path.basename(str(x)) == img_name if pd.notna(x) else False)]
        is_existing = len(matches) > 0
        
        if is_existing and pd.notna(matches["predicted_category"].values[0]):
            print(f"-> Skipping {img_name} (Already tagged)")
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"Processing: {img_name}")
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Extract tags using GPU-accelerated CLIP
            cat = get_best_match(image, categories)
            col = get_best_match(image, colors)
            mat = get_best_match(image, materials)
            sty = get_best_match(image, styles)

            # Generate smart fallbacks for the title if it's completely blank
            fallback_title = os.path.splitext(img_name)[0].replace("_", " ").title()

            if is_existing:
                match_idx = matches.index[0]
                df.loc[match_idx, ["predicted_category", "color", "material", "style"]] = [cat, col, mat, sty]
                
                # If title is missing, fill fallback
                if pd.isna(df.loc[match_idx, "title"]):
                    df.loc[match_idx, "title"] = fallback_title
            else:
                # Build fresh row structure
                next_id = f"prod_{len(df) + 1}"
                new_row = pd.DataFrame([{
                    "product_id": next_id,
                    "title": fallback_title,
                    "image_path": os.path.join(".", "data", "data_images", img_name),
                    "predicted_category": cat,
                    "color": col,
                    "material": mat,
                    "style": sty
                }])
                df = pd.concat([df, new_row], ignore_index=True)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    # Save updates back directly to your drive path
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSuccessfully batch tagged and saved results to {CSV_PATH}!")

if __name__ == "__main__":
    process_batch_tagging()