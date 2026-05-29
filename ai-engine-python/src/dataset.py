import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ECommerceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Resolve image path
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {img_name}. Error: {e}")

        product_title = self.metadata.iloc[idx, 1]
        category = self.metadata.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': product_title,
            'category': category
        }

def get_clip_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])