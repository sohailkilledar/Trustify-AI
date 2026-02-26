import cv2
import torch
import numpy as np
from PIL import Image, ImageChops
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

artifact_model_name = "umm-maybe/AI-image-detector"
artifact_processor = AutoImageProcessor.from_pretrained(artifact_model_name)
artifact_model = AutoModelForImageClassification.from_pretrained(artifact_model_name).to(device).eval()

clip_pipeline = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=0 if torch.cuda.is_available() else -1
)

clip_labels = [
    "real photograph", "camera photo", "human photo",
    "AI generated image", "digital art", "synthetic image",
    "deepfake", "AI portrait", "machine generated"
]

def get_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    resaved_path = 'temp_resaved.jpg'
    original.save(resaved_path, 'JPEG', quality=quality)
    resaved = Image.open(resaved_path)
    ela_im = ImageChops.difference(original, resaved)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageChops.constant(ela_im, scale)
    return np.array(ela_im).mean()

def analyze_frequency_variance(image_path):
    img = cv2.imread(image_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return np.var(magnitude_spectrum)

def check_image_hybrid(image_path):
    try:
        clip_result = clip_pipeline(image_path, candidate_labels=clip_labels)
        clip_ai_score = sum([x['score'] for x in clip_result if any(kw in x['label'].lower() for kw in ['ai', 'synthetic', 'digital', 'generated'])])
        clip_real_score = sum([x['score'] for x in clip_result if any(kw in x['label'].lower() for kw in ['real', 'photo', 'camera'])])
        clip_conf = clip_ai_score / (clip_ai_score + clip_real_score + 1e-6)
    except:
        clip_conf = 0.5

    try:
        img = Image.open(image_path).convert('RGB')
        inputs = artifact_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = artifact_model(**inputs)
            artifact_ai_prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
    except:
        artifact_ai_prob = 0.5

    try:
        ela_val = get_ela(image_path)
        freq_var = analyze_frequency_variance(image_path)
        forensic_signal = (min(ela_val / 15.0, 1.0) * 0.5) + (min(freq_var / 250.0, 1.0) * 0.5)
    except:
        forensic_signal = 0.5

    combined_score = (clip_conf * 0.3) + (artifact_ai_prob * 0.5) + (forensic_signal * 0.2)
    combined_score = min(0.99, max(0.01, combined_score))

    if combined_score > 0.82:
        verdict = "AI Generated / Deepfake"
    elif combined_score > 0.55:
        verdict = "Suspicious Content"
    else:
        verdict = "Authentic Image"

    return {
        "verdict": verdict,
        "ai_probability": round(combined_score, 4),
        "confidence_metrics": {
            "semantic": round(clip_conf, 4),
            "texture_artifact": round(artifact_ai_prob, 4),
            "forensic_trace": round(forensic_signal, 4)
        }
    }

if __name__ == "__main__":
    file_path = "target_image.jpg"
    print(check_image_hybrid(file_path))
