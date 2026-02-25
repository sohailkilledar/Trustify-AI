import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import os
import time

def test_gpu_acceleration():
    try:
        print("🎮 Testing GPU acceleration...")
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 GPU device: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test both CPU and GPU performance
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test image
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        test_image.save("gpu_test_image.jpg")
        
        # Test with GPU acceleration
        print(f"🚀 Testing with device: {device}")
        start_time = time.time()
        
        classifier = pipeline("zero-shot-image-classification", 
                           model="openai/clip-vit-base-patch32",
                           device=device,
                           torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        
        candidate_labels = ["real photograph", "AI generated image", "computer generated artwork"]
        result = classifier("gpu_test_image.jpg", candidate_labels=candidate_labels)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⚡ Processing time: {processing_time:.2f} seconds")
        print(f"📄 Result: {result}")
        
        # Show memory usage if GPU
        if torch.cuda.is_available():
            print(f"💾 GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"💾 GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("gpu_test_image.jpg"):
            os.remove("gpu_test_image.jpg")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_gpu_acceleration()
