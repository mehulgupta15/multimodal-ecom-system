import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel

class CLIPEngine(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super(CLIPEngine, self).__init__()
        print(f"Initializing CLIP Engine with weights: {model_name}...")
        
        # 1. Load the official Hugging Face CLIP components
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
        # 2. Determine device acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model successfully loaded and mapped to device: {self.device.type.upper()}")

    def tokenize_text(self, text_list: list) -> dict:
        """
        Tokenizes raw text strings into fixed-length matrices of 77 tokens.
        """
        # CLIP's native max sequence length is exactly 77 tokens
        tokens = self.tokenizer(
            text_list, 
            padding="max_length", 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        # Move token tensors to the active device
        return {k: v.to(self.device) for k, v in tokens.items()}

    def extract_image_features(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Passes preprocessed image tensors through the Vision Encoder.
        """
        image_tensors = image_tensors.to(self.device)
        with torch.no_grad():
            # 1. Extract features using the model
            outputs = self.model.get_image_features(pixel_values=image_tensors)
            
            # 2. Extract the raw tensor if wrapped in a HF output object
            if hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            else:
                image_features = outputs  # It's already a raw tensor
                
            # 3. Safely apply L2 normalization
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def extract_text_features(self, text_list: list) -> torch.Tensor:
        """
        Tokenizes and passes text strings through the Text Encoder.
        """
        tokenized_inputs = self.tokenize_text(text_list)
        with torch.no_grad():
            # 1. Extract features using the model
            outputs = self.model.get_text_features(**tokenized_inputs)
            
            # 2. Extract the raw tensor if wrapped in a HF output object
            if hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            else:
                text_features = outputs  # It's already a raw tensor
                
            # 3. Safely apply L2 normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features