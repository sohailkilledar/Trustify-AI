import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "prithivMLmods/Deep-Fake-Detector-Model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.to(device)
model.eval()

mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])

def check_video_authenticity(path, max_frames=50):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return {
            "type": "Video Authenticity Assessment",
            "verdict": "Error Opening Video / Cannot Analyze",
            "confidence": 0,
            "deepfake_probability": 0.0
        }

    frame_count = 0
    predictions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)
    current_frame = 0

    while cap.isOpened() and frame_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                h, w, _ = rgb.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face = rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                inputs = processor(images=face, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    deepfake_prob = probs[0][1].item()
                predictions.append(deepfake_prob)

        frame_count += 1
        current_frame += frame_step

    cap.release()

    if len(predictions) == 0:
        return {
            "type": "Video Authenticity Assessment",
            "verdict": "No Detectable Faces for Verification",
            "confidence": 0,
            "deepfake_probability": 0.0
        }

    avg_score = float(np.mean(predictions))

    if avg_score > 0.65:
        verdict = "Manipulated Content Detected / Likely Deepfake"
    elif avg_score < 0.45:
        verdict = "Authentic Content / Likely Real Video"
    else:
        verdict = "Indeterminate / Requires Manual Verification"

    confidence = int(min(100, abs(avg_score - 0.5) * 200))

    return {
        "type": "Video Authenticity Assessment",
        "verdict": verdict,
        "confidence": confidence,
        "deepfake_probability": avg_score
    }