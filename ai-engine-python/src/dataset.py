import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ECommerceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        raw_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter metadata to keep ONLY rows where the physical image file actually exists on disk
        valid_rows = []
        for idx, row in raw_df.iterrows():
            raw_path = str(row.iloc[3])
            fname = os.path.basename(raw_path)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fname = f"{fname}.jpg"
            target_file = os.path.join(img_dir, fname)
            if os.path.exists(target_file) and os.path.getsize(target_file) > 100:
                valid_rows.append(row)
                
        if len(valid_rows) > 0:
            self.metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
        else:
            self.metadata = raw_df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Grab raw data from targeted schema indices
        # Col 0 = product_id, Col 1 = title, Col 3 = image_path
        product_id = str(self.metadata.iloc[idx, 0])
        raw_img_path = str(self.metadata.iloc[idx, 3])
        product_title = str(self.metadata.iloc[idx, 1])

        # 2. Path Sanitation Engine
        # Extract just the raw filename (e.g., 'prod_1.jpg') to clear slash mismatches
        filename = os.path.basename(raw_img_path)
        
        # Security check: force extension if the file row dropped it
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = f"{filename}.jpg"

        # Construct absolute targeted path on your E: drive
        img_path = os.path.join(self.img_dir, filename)

        # 3. Secure File Ingestion Block (Fault Tolerant)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to neutral 224x224 RGB image if a single file is corrupted or unreadable
            image = Image.new('RGB', (224, 224), color=(240, 240, 240))

        # 4. Image Preprocessing Matrix Transformation
        if self.transform:
            image = self.transform(image)

        # 5. Production Return Contract
        # We pass back the image tensor and the descriptive string label for your FAISS ledger.
        # You can return 'product_title' here so your search results print real product names!
        return image, product_title

def get_clip_transforms():
    """Standard normalization transforms for CLIP input images"""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])