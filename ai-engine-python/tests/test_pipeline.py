import sys
import os
from torch.utils.data import DataLoader

# Ensure the script can see 'src' folder inside ai-engine-python
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.dataset import ECommerceDataset, get_clip_transforms

def verify_pipeline():
    print("🚀 Initializing E-Commerce Data Pipeline...")
    
    # Define paths explicitly relative to this workspace
    csv_path = os.path.join(BASE_DIR, "data", "metadata.csv")
    img_dir = os.path.join(BASE_DIR, "data", "mock_images")
    
    # 1. Initialize dataset
    dataset = ECommerceDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transform=get_clip_transforms()
    )
    
    print(f"✅ Dataset loaded successfully. Total items: {len(dataset)}")
    
    # 2. Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # 3. Test a batch run
    for batch in dataloader:
        images = batch['image']
        texts = batch['text']
        
        print("\n--- Tensor Verification ---")
        print(f"Images tensor shape : {images.shape}")  # Expected: [batch_size, 3, 224, 224]
        print(f"Batch texts         : {texts}")
        
        assert images.shape == (len(texts), 3, 224, 224), "❌ Image tensor shape mismatch!"
        print("\n🎉 Success! The PyTorch pipeline is structurally sound and ready for CLIP.")
        break

if __name__ == "__main__":
    verify_pipeline()