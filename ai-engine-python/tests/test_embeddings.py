import sys
import os
import torch
from torch.utils.data import DataLoader
# 1. Import torchvision transforms
from torchvision import transforms 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import ECommerceDataset
from src.model import CLIPEngine

def test_embedding_pipeline():
    print("\n=== Launching Day 4 Integration Test ===")
    
    csv_path = r"E:\multimodal-ecom-system\ai-engine-python\data\metadata.csv"
    image_dir = r"E:\multimodal-ecom-system\ai-engine-python\data\mock_images"
    
    # 2. Define the evaluation transform pipeline for CLIP
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # CLIP expects 224x224 images
        transforms.ToTensor(),          # Converts PIL Image to torch.Tensor [C, H, W]
        # Standard ImageNet/CLIP normalization
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # 3. Pass the transform into your dataset (matching your parameter: transform=...)
    dataset = ECommerceDataset(
        csv_file=csv_path, 
        img_dir=image_dir, 
        transform=clip_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Fetch a single batch
    # Fetch a single batch
    batch = next(iter(dataloader))
    images = batch['image']
    titles = batch['text']  # ✨ Changed 'title' to 'text' to match your dataset!
    
    print(f"Successfully loaded batch from Day 3 pipeline.")
    print(f"↳ Raw Image Tensor Shape: {images.shape}")
    print(f"↳ Batch Titles: {titles}")
    
    # 4. Instantiate Day 4 model engine
    engine = CLIPEngine()
    
    # 5. Extract Text and Image Embeddings
    print("\nPassing assets through neural network pathways...")
    image_embeddings = engine.extract_image_features(images)
    text_embeddings = engine.extract_text_features(titles)
    
    print("\n=== Vector Embedding Footprints ===")
    print(f"↳ Image Embedding Shape: {image_embeddings.shape}")
    print(f"↳ Text Embedding Shape:  {text_embeddings.shape}")
    
    assert image_embeddings.shape == (2, 512), f"Expected shape (2, 512), got {image_embeddings.shape}"
    assert text_embeddings.shape == (2, 512), f"Expected shape (2, 512), got {text_embeddings.shape}"
    
    print("\n✅ SUCCESS: Raw data successfully converted into shared 512-D vector space!")

if __name__ == "__main__":
    test_embedding_pipeline()