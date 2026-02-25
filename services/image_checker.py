from PIL import Image
import imagehash
import exifread
import os
import torch
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def detect_ai_generated(image_path):
    """Advanced AI detection with GPU acceleration and focused analysis"""
    try:
        from transformers import pipeline
        
        # Force GPU usage
        device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
        print(f"🚀 Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
        
        # Use only the most effective methods
        results = {}
        
        # 1. CLIP Zero-Shot Classification (most reliable)
        results['clip_analysis'] = analyze_with_clip(image_path, device)
        
        # 2. Simple but effective visual analysis
        results['visual_analysis'] = analyze_visual_patterns(image_path)
        
        # Combine with simpler scoring
        final_result = combine_detections_simplified(results)
        
        return final_result
        
    except Exception as e:
        return {
            "ai_generated": False,
            "confidence": 0,
            "method": "AI Detection Error",
            "error": True,
            "error_message": str(e)
        }

def analyze_with_clip(image_path, device):
    """CLIP analysis for AI vs real classification"""
    try:
        classifier = pipeline("zero-shot-image-classification", 
                           model="openai/clip-vit-base-patch32",
                           device=device,
                           torch_dtype=torch.float16 if device == 0 else torch.float32)
        
        # Focused AI detection labels
        candidate_labels = [
            "real photograph", "camera photo", "human photo",
            "AI generated image", "computer generated artwork", "digital art",
            "synthetic image", "artificial intelligence generated", "machine generated",
            "3D render", "computer graphics", "digital illustration",
            "deepfake", "AI portrait", "generated face", "AI art"
        ]
        
        result = classifier(image_path, candidate_labels=candidate_labels)
        
        # Calculate AI vs real scores
        ai_score = 0
        real_score = 0
        ai_indicators = []
        real_indicators = []
        
        for item in result:
            label = item['label'].lower()
            score = item['score']
            
            # AI indicators (more comprehensive)
            if any(term in label for term in ['ai', 'computer', 'synthetic', 'artificial', 'machine', 'digital', 'generated', 'render', 'graphics', 'illustration', 'deepfake', 'art']):
                ai_score += score
                ai_indicators.append(f"{label}: {score:.3f}")
            # Real indicators
            elif any(term in label for term in ['real', 'human', 'camera', 'photograph', 'photo']):
                real_score += score
                real_indicators.append(f"{label}: {score:.3f}")
        
        # More sensitive AI detection
        is_ai = ai_score > real_score * 0.8  # Lower threshold for AI detection
        
        return {
            "ai_score": ai_score,
            "real_score": real_score,
            "verdict": "AI" if is_ai else "Real",
            "confidence": max(ai_score, real_score),
            "ai_indicators": ai_indicators,
            "real_indicators": real_indicators,
            "top_predictions": result[:5]
        }
        
    except Exception as e:
        return {"error": str(e), "ai_score": 0, "real_score": 0}

def analyze_visual_patterns(image_path):
    """Simple but effective visual analysis for AI detection"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Simple checks that work well for AI detection
        analysis = {}
        
        # 1. Check image dimensions (AI images often have standard sizes)
        h, w = img_array.shape[:2]
        standard_sizes = [(512, 512), (1024, 1024), (256, 256), (768, 768)]
        is_standard_size = any((h, w) == size or (w, h) == size for size in standard_sizes)
        analysis['standard_size'] = is_standard_size
        
        # 2. Check for perfect symmetry (common in AI images)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            if left_half.shape == right_half.shape:
                symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                analysis['symmetry'] = symmetry
            else:
                analysis['symmetry'] = 0
        else:
            analysis['symmetry'] = 0
        
        # 3. Color distribution (AI images often have different color patterns)
        if len(img_array.shape) == 3:
            color_std = np.std(img_array, axis=(0, 1))
            analysis['color_uniformity'] = 1.0 - np.mean(color_std) / 128.0
        else:
            analysis['color_uniformity'] = 0
        
        # Calculate AI probability based on visual patterns
        ai_score = 0
        if analysis['standard_size']:
            ai_score += 0.3
        if analysis['symmetry'] > 0.8:
            ai_score += 0.4
        if analysis['color_uniformity'] > 0.6:
            ai_score += 0.3
        
        return {
            "ai_score": ai_score,
            "analysis": analysis,
            "verdict": "AI" if ai_score > 0.5 else "Real"
        }
        
    except Exception as e:
        return {"error": str(e), "ai_score": 0}

def combine_detections_simplified(results):
    """Simplified combination focusing on CLIP results"""
    
    clip_result = results.get('clip_analysis', {})
    visual_result = results.get('visual_analysis', {})
    
    # Primary weight on CLIP (most reliable)
    clip_ai_score = clip_result.get('ai_score', 0)
    clip_real_score = clip_result.get('real_score', 0)
    
    # Secondary weight on visual analysis
    visual_ai_score = visual_result.get('ai_score', 0)
    
    # Combined scoring
    total_ai_score = (clip_ai_score * 0.8) + (visual_ai_score * 0.2)
    total_real_score = clip_real_score * 0.8
    
    # Determine verdict
    is_ai = total_ai_score > total_real_score * 0.7  # Sensitive threshold
    confidence = min(95, int(max(total_ai_score, total_real_score) * 100))
    
    return {
        "ai_generated": is_ai,
        "confidence": confidence,
        "method": "Advanced AI Detection (GPU Accelerated)",
        "total_score": total_ai_score,
        "clip_ai_score": clip_ai_score,
        "clip_real_score": clip_real_score,
        "visual_ai_score": visual_ai_score,
        "ai_indicators": clip_result.get('ai_indicators', []),
        "real_indicators": clip_result.get('real_indicators', []),
        "top_predictions": clip_result.get('top_predictions', []),
        "device_used": "GPU (CUDA)" if torch.cuda.is_available() else "CPU",
        "error": False
    }

def analyze_for_ai_indicators(result):
    """Analyze Hugging Face result for AI indicators"""
    if isinstance(result, list) and len(result) > 0:
        # Look for AI-related labels in the predictions
        ai_indicators = [
            'illustration', 'painting', 'art', 'digital', 'computer', 
            'graphic', 'render', 'synthetic', 'cartoon', 'drawing',
            'sketch', 'abstract', 'design', 'pattern'
        ]
        
        ai_score = 0
        detected_labels = []
        
        for prediction in result[:5]:  # Top 5 predictions
            label = prediction.get('label', '').lower()
            score = prediction.get('score', 0)
            
            detected_labels.append(f"{prediction.get('label', '')}: {score:.2f}")
            
            # Check if label indicates AI generation
            if any(indicator in label for indicator in ai_indicators):
                ai_score += score
        
        # Also check for non-photographic labels
        non_photo_indicators = ['animal', 'object', 'plant', 'food', 'building']
        for prediction in result[:5]:
            label = prediction.get('label', '').lower()
            score = prediction.get('score', 0)
            
            if any(indicator in label for indicator in non_photo_indicators):
                ai_score += score * 0.5  # Lower weight for non-AI but non-photo
        
        # Determine if AI generated
        is_ai = ai_score > 0.3
        confidence = min(95, int(ai_score * 100))
        
        return {
            "ai_generated": is_ai,
            "confidence": confidence,
            "method": "Hugging Face ViT",
            "ai_score": ai_score,
            "detected_labels": detected_labels,
            "error": False
        }
    
    else:
        return {
            "ai_generated": False,
            "confidence": 0,
            "method": "Invalid response",
            "error": True
        }

def check_image(path):
    """Enhanced image checker with AI detection"""
    
    # Original EXIF and pHash analysis
    img = Image.open(path)
    hash_value = imagehash.phash(img)

    exif_data = {}
    with open(path, "rb") as f:
        tags = exifread.process_file(f, stop_tag="DateTimeOriginal")
        for tag in tags:
            exif_data[tag] = str(tags[tag])

    # NEW: AI Detection
    ai_result = detect_ai_generated(path)
    
    # Determine verdict based on AI detection
    if ai_result["ai_generated"] and not ai_result.get("error"):
        verdict = "AI Generated"
        confidence = max(ai_result["confidence"], 75)
        details = f"AI Detection: {ai_result['method']} (Confidence: {ai_result['confidence']}%)"
        
        # Add detected labels
        if ai_result.get("detected_labels"):
            details += f" | Labels: {', '.join(ai_result['detected_labels'][:3])}"
            
    else:
        # Original logic for real/manipulated images
        manipulated = len(exif_data) == 0
        if manipulated:
            verdict = "Possibly Manipulated"
            confidence = 65
            details = f"pHash: {hash_value} - No EXIF data found"
        else:
            verdict = "Likely Authentic"
            confidence = 80
            details = f"pHash: {hash_value} - Real photograph"
        
        # Add AI detection info if it failed
        if ai_result.get("error"):
            details += f" | AI Detection: {ai_result['method']}"
    
    return {
        "type": "Enhanced Image Verification",
        "verdict": verdict,
        "confidence": confidence,
        "details": details,
        "ai_detection": ai_result,
        "exif_data": exif_data,
        "phash": str(hash_value)
    }
