from transformers import pipeline
from PIL import Image
import numpy as np
import os

def test_clip_ai_detection():
    try:
        print("🤖 Testing CLIP AI detection...")
        
        # Create a test image that looks more like AI art
        # Create a colorful abstract pattern (more likely to be detected as AI)
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some patterns that might look like AI art
        test_array[50:100, 50:100] = [255, 0, 128]  # Pink square
        test_array[120:170, 120:170] = [0, 255, 128]  # Green square
        test_array = np.clip(test_array * 1.5, 0, 255).astype(np.uint8)  # Brighten
        
        test_image = Image.fromarray(test_array)
        test_image.save("test_ai_image.jpg")
        
        # Use CLIP for AI detection
        classifier = pipeline("zero-shot-image-classification", 
                           model="openai/clip-vit-base-patch32")
        
        # Define labels for AI vs real
        candidate_labels = [
            "real photograph", 
            "AI generated image", 
            "computer generated artwork",
            "digital art",
            "human photo",
            "synthetic image"
        ]
        
        print("📤 Analyzing test image for AI indicators...")
        result = classifier("test_ai_image.jpg", candidate_labels=candidate_labels)
        
        print("✅ CLIP working!")
        print(f"📄 Result: {result}")
        
        # Analyze results
        ai_score = sum(item['score'] for item in result 
                      if any(ai_term in item['label'].lower() 
                            for ai_term in ['ai', 'computer', 'synthetic', 'digital']))
        
        real_score = sum(item['score'] for item in result 
                        if any(real_term in item['label'].lower() 
                              for real_term in ['real', 'human', 'photograph']))
        
        print(f"🤖 AI Score: {ai_score:.3f}")
        print(f"📷 Real Score: {real_score:.3f}")
        print(f"🎯 Verdict: {'AI Generated' if ai_score > real_score else 'Real Image'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_ai_image.jpg"):
            os.remove("test_ai_image.jpg")

if __name__ == "__main__":
    test_clip_ai_detection()
