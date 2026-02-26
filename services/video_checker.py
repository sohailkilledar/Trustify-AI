import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "prithivMLmods/Deep-Fake-Detector-Model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()

mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.7, 0.8, 0.8])

def check_video_authenticity(path, max_frames=60):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"verdict": "Error", "confidence": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    predictions = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                x1, y1 = max(0, x1 - int(w * 0.15)), max(0, y1 - int(h * 0.15))
                x2, y2 = min(rgb_frame.shape[1], x2 + int(w * 0.15)), min(rgb_frame.shape[0], y2 + int(h * 0.15))
                
                face = rgb_frame[int(y1):int(y2), int(x1):int(x2)]
                if face.size == 0: continue

                inputs = processor(images=face, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    fake_prob = probs[0][1].item()
                predictions.append(fake_prob)

    cap.release()

    if not predictions:
        return {"verdict": "No faces detected", "score": 0}

    avg_score = np.mean(predictions)
    max_score = np.max(predictions)
    std_dev = np.std(predictions)
    final_score = (avg_score * 0.6) + (max_score * 0.4)

    if final_score > 0.70:
        verdict = "AI-Generated / Deepfake"
    elif final_score < 0.30:
        verdict = "Authentic / Real"
    else:
        verdict = "Inconclusive"

    return {
        "verdict": verdict,
        "probability": round(final_score, 4),
        "instability": round(std_dev, 4),
        "confidence": f"{round((1 - std_dev) * 100, 2)}%"
    }

if __name__ == "__main__":
    video_path = "input_video.mp4"
    print(check_video_authenticity(video_path))
