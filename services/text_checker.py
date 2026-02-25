import requests
import os
from dotenv import load_dotenv
from services.gemini_checker import GeminiChecker

# Load environment variables
load_dotenv()

API_KEY = "d64d65bdda8f4a78b2573128c24fe42a"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def fetch_news_simple(query):
    """Simple news fetching without ML processing"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": 5,
        "apiKey": API_KEY
    }
    try:
        r = requests.get(url, params=params)
        articles = r.json().get("articles", [])
        
        # Simple keyword-based analysis
        support_count = 0
        refute_count = 0
        sources = []
        
        for article in articles:
            content = (article["title"] or "") + " " + (article["description"] or "")
            source_name = article["source"]["name"]
            sources.append(source_name)
            
            content_lower = content.lower()
            if any(word in content_lower for word in ["confirms", "reports", "says", "true", "accurate"]):
                support_count += 1
            elif any(word in content_lower for word in ["denies", "debunks", "false", "fake"]):
                refute_count += 1
        
        if support_count > refute_count and support_count > 0:
            verdict = "Likely True"
            confidence = min(60 + support_count * 10, 85)
        elif refute_count > support_count and refute_count > 0:
            verdict = "Likely False"
            confidence = min(60 + refute_count * 10, 85)
        else:
            verdict = "Unverified"
            confidence = 40
            
        return {
            "type": "News Source Analysis",
            "verdict": verdict,
            "confidence": confidence,
            "details": f"Based on {len(articles)} news articles",
            "sources": list(set(sources))
        }
        
    except Exception as e:
        return {
            "type": "News Source Analysis",
            "verdict": "Unverified",
            "confidence": 20,
            "details": f"Error fetching news: {str(e)}",
            "sources": []
        }

def check_text(user_text):
    """Dual verification using News API + Gemini API"""
    
    # 1. News Analysis (always works)
    news_result = fetch_news_simple(user_text)
    
    # 2. Gemini AI Analysis (with error handling)
    gemini_result = None
    try:
        gemini_checker = GeminiChecker(GEMINI_API_KEY)
        gemini_result = gemini_checker.analyze_claim_reasoning(user_text, news_result.get("sources", []))
    except Exception as e:
        # Gemini API failed (quota exceeded, network error, etc.)
        gemini_result = {
            "verdict": "Unverified",
            "confidence": 30,
            "details": f"Gemini API unavailable: {str(e)}",
            "error": True
        }
    
    # Combine results
    return combine_dual_verification(news_result, gemini_result)

def combine_dual_verification(news_result, gemini_result):
    """Combine news and Gemini results"""
    
    # Convert verdicts to numeric scores
    verdict_scores = {
        "Likely True": 0.8,
        "True": 0.8,
        "Likely False": 0.2,
        "False": 0.2,
        "Mixed/Unverified": 0.5,
        "Unverified": 0.4
    }
    
    news_score = verdict_scores.get(news_result["verdict"], 0.4)
    
    # Handle Gemini API failure
    if gemini_result.get("error"):
        # If Gemini failed, rely more heavily on news but be conservative
        final_verdict = news_result["verdict"]
        combined_confidence = max(40, news_result["confidence"] - 10)  # Reduce confidence slightly
        combined_score = news_score
        
        return {
            "type": "Dual Verification System (News + Gemini AI)",
            "verdict": final_verdict,
            "confidence": combined_confidence,
            "details": f"News analysis only (Gemini API unavailable)",
            "news_verification": news_result,
            "gemini_reasoning": gemini_result,
            "combined_score": round(combined_score, 2),
            "api_status": "news_only"
        }
    
    # Normal dual verification when both APIs work
    gemini_score = verdict_scores.get(gemini_result["verdict"], 0.4)
    
    # Weight the results (Gemini gets more weight for reasoning)
    news_confidence = news_result["confidence"] / 100
    gemini_confidence = gemini_result["confidence"] / 100
    
    combined_score = (news_score * 0.4 * news_confidence) + (gemini_score * 0.6 * gemini_confidence)
    
    # Determine final verdict
    if combined_score >= 0.65:
        final_verdict = "Likely True"
    elif combined_score <= 0.35:
        final_verdict = "Likely False"
    else:
        final_verdict = "Mixed/Unverified"
    
    # Calculate combined confidence with boost for agreement
    base_confidence = int((news_result["confidence"] + gemini_result["confidence"]) / 2)
    
    # Boost confidence if both agree
    if (news_result["verdict"] == gemini_result["verdict"] and 
        "Unverified" not in news_result["verdict"]):
        combined_confidence = min(90, base_confidence + 15)
    else:
        combined_confidence = base_confidence
    
    return {
        "type": "Dual Verification System (News + Gemini AI)",
        "verdict": final_verdict,
        "confidence": combined_confidence,
        "details": f"Combined analysis from news sources and AI reasoning",
        "news_verification": news_result,
        "gemini_reasoning": gemini_result,
        "combined_score": round(combined_score, 2),
        "api_status": "dual_working"
    }
