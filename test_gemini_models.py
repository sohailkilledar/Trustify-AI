import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(dotenv_path='.env')

def list_available_models():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Gemini API key not configured")
            return
        
        genai.configure(api_key=api_key)
        
        # List available models
        print("📋 Available Gemini Models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
            else:
                print(f"❌ {model.name} (no generateContent)")
                
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    list_available_models()
