import os
import requests
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test Hugging Face API
def test_huggingface_api():
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    print(f"API Key: {api_key[:20]}..." if api_key else "No API key found")
    
    if not api_key:
        print("❌ No API key configured")
        return False
    
    # Test with a simple image (create a test image)
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_image.save("test_image.jpg")
        
        # Convert to base64
        with open("test_image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        print("📤 Sending test request to Hugging Face API...")
        
        # Test API call
        API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {"inputs": image_data}
        response = requests.post(API_URL, headers=headers, json=payload)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📝 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API working!")
            print(f"📄 Response: {result}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Clean up test image
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")

if __name__ == "__main__":
    print("🔍 Testing Hugging Face API...")
    print("-" * 50)
    test_huggingface_api()
