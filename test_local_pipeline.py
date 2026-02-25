from transformers import pipeline
from PIL import Image
import numpy as np
import os

def test_local_pipeline():
    try:
        print("🤖 Testing local Hugging Face pipeline...")
        
        # Create a test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_image.save("test_image.jpg")
        
        # Use local pipeline
        classifier = pipeline("image-classification", 
                           model="microsoft/resnet-50")
        
        print("📤 Classifying test image...")
        result = classifier("test_image.jpg")
        
        print("✅ Pipeline working!")
        print(f"📄 Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")

if __name__ == "__main__":
    test_local_pipeline()
