import torch
from transformers import CLIPProcessor, CLIPModel

print("--- Starting Environment Smoke Test ---")

# Verify hardware acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target Hardware Acceleration Device: {device.upper()}")

try:
    print("Fetching and initializing CLIP weights from Hugging Face Hub...")
    model_id = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    print("\n[SUCCESS] Model and processor successfully loaded into memory!")
    print("Environment Verification: PASSED")

except Exception as e:
    print("\n[ERROR] Something went wrong during initialization:")
    print(str(e))

print("---------------------------------------")