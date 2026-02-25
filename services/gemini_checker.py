import os
import google.generativeai as genai
from typing import Dict, List, Optional

class GeminiChecker:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key and self.api_key != "YOUR_GEMINI_API_KEY":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.available = True
        else:
            self.available = False
    
    def analyze_claim_reasoning(self, claim: str, context: List[str] = None) -> Dict:
        """
        Analyze claim using Gemini's reasoning capabilities
        """
        if not self.available:
            return self._mock_gemini_analysis(claim, context)
        
        try:
            # Construct the prompt for fact-checking
            prompt = self._build_fact_check_prompt(claim, context)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse the response
            analysis = self._parse_gemini_response(response.text)
            
            return analysis
            
        except Exception as e:
            return {
                "verdict": "Unverified",
                "confidence": 30,
                "details": f"Gemini analysis failed: {str(e)}",
                "error": True
            }
    
    def _build_fact_check_prompt(self, claim: str, context: List[str] = None) -> str:
        """Build a comprehensive prompt for fact-checking"""
        
        base_prompt = f"""You are an expert fact-checker and misinformation analyst. Analyze the following claim objectively:

CLAIM: "{claim}"

Please provide a detailed analysis covering:
1. Factual accuracy of the claim
2. Logical consistency
3. Potential for misinformation
4. Confidence level in your assessment

Respond in this exact format:
VERDICT: [Likely True/Likely False/Mixed/Unverified]
CONFIDENCE: [0-100]
REASONING: [Detailed explanation]
EVIDENCE: [What evidence supports this conclusion]

Be objective, evidence-based, and avoid speculation. Focus on verifiable facts."""
        
        if context:
            context_text = "\n".join([f"- {ctx}" for ctx in context[:5]])
            base_prompt += f"\n\nADDITIONAL CONTEXT:\n{context_text}\n\nConsider this context in your analysis."
        
        return base_prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's response into structured format"""
        
        lines = response_text.strip().split('\n')
        
        verdict = "Mixed/Unverified"
        confidence = 50
        reasoning = ""
        evidence = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = int(line.replace("CONFIDENCE:", "").strip())
                except:
                    confidence = 50
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("EVIDENCE:"):
                evidence = line.replace("EVIDENCE:", "").strip()
        
        # Normalize verdict
        if "true" in verdict.lower():
            verdict = "Likely True"
        elif "false" in verdict.lower():
            verdict = "Likely False"
        elif "mixed" in verdict.lower() or "unverified" in verdict.lower():
            verdict = "Mixed/Unverified"
        
        return {
            "verdict": verdict,
            "confidence": min(95, max(20, confidence)),
            "details": f"Gemini AI Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}",
            "evidence": evidence[:150] + "..." if len(evidence) > 150 else evidence,
            "full_reasoning": reasoning
        }
    
    def _mock_gemini_analysis(self, claim: str, context: List[str] = None) -> Dict:
        """Mock analysis when no API key available"""
        claim_lower = claim.lower()
        
        # Simple heuristic-based mock analysis
        sensational_words = ["shocking", "incredible", "amazing", "breakthrough", "miracle"]
        conspiracy_words = ["hoax", "conspiracy", "cover-up", "secret", "hidden"]
        science_words = ["study", "research", "scientists", "evidence", "data"]
        
        sensational_score = sum(1 for word in sensational_words if word in claim_lower)
        conspiracy_score = sum(1 for word in conspiracy_words if word in claim_lower)
        science_score = sum(1 for word in science_words if word in claim_lower)
        
        if conspiracy_score > 2:
            return {
                "verdict": "Likely False",
                "confidence": 65,
                "details": "Mock Gemini: Contains conspiracy indicators",
                "mock": True
            }
        elif sensational_score > 2 and science_score == 0:
            return {
                "verdict": "Mixed/Unverified",
                "confidence": 45,
                "details": "Mock Gemini: Sensational language without scientific backing",
                "mock": True
            }
        elif science_score > 1:
            return {
                "verdict": "Likely True",
                "confidence": 60,
                "details": "Mock Gemini: Contains scientific indicators",
                "mock": True
            }
        else:
            return {
                "verdict": "Mixed/Unverified",
                "confidence": 50,
                "details": "Mock Gemini: Insufficient indicators for analysis",
                "mock": True
            }
