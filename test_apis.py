import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Test Gemini API
def test_gemini():
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        
        print(f"Debug: Gemini API key from env: {api_key[:20]}..." if api_key else "Debug: No Gemini API key found")
        
        if not api_key or api_key == "YOUR_GEMINI_API_KEY":
            print("❌ Gemini API key not configured")
            return False
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Hello, this is a test message. Respond with 'Gemini working!'")
        
        if "Gemini working!" in response.text:
            print("✅ Gemini API working!")
            return True
        else:
            print(f"✅ Gemini API working! Response: {response.text[:50]}...")
            return True
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

# Test News API
def test_news_api():
    try:
        import requests
        api_key = os.getenv("NEWS_API_KEY")
        
        print(f"Debug: News API key from env: {api_key[:20]}..." if api_key else "Debug: No News API key found")
        
        if not api_key:
            print("❌ News API key not configured")
            return False
            
        # Test with a simple search
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "test",
            "language": "en",
            "pageSize": 1,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"✅ News API working! Found {len(articles)} articles")
            return True
        else:
            print(f"❌ News API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ News API error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Dual Verification APIs...")
    print("-" * 50)
    
    gemini_ok = test_gemini()
    print()
    news_ok = test_news_api()
    
    print("-" * 50)
    
    if gemini_ok and news_ok:
        print("🎉 All APIs ready! Dual verification system operational.")
    elif gemini_ok:
        print("✅ Gemini ready! News API needs configuration.")
    elif news_ok:
        print("✅ News API ready! Gemini needs configuration.")
    else:
        print("⚠️ No APIs are working. Check your API keys and try again.")
    
    print("\n📋 Status:")
    print(f"   Gemini API: {'✅' if gemini_ok else '❌'}")
    print(f"   News API: {'✅' if news_ok else '❌'}")
