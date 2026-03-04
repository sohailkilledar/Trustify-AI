import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from collections import defaultdict
import warnings
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME_FACE = "prithivMLmods/Deep-Fake-Detector-Model"
MAX_FRAMES_TO_ANALYZE = 60
MTCNN_THRESHOLDS = [0.7, 0.8, 0.8]

# Thresholds
FACE_FAKE_THRESHOLD = 0.6
SUSPICIOUS_THRESHOLD = 0.35
CINEMATIC_BAR_THRESHOLD = 10
TEMPORAL_VARIANCE_THRESHOLD = 0.25
BLUR_LOW_THRESHOLD = 50

print(f"[*] Initializing Industry-Level Deepfake Forensic System on {DEVICE}...")

processor_face = AutoImageProcessor.from_pretrained(MODEL_NAME_FACE)
model_face = AutoModelForImageClassification.from_pretrained(MODEL_NAME_FACE).to(DEVICE)
model_face.eval()
ID2LABEL_FACE = model_face.config.id2label
FAKE_LABEL_INDEX = next((i for i, label in ID2LABEL_FACE.items() if "fake" in label.lower()), 0)

mtcnn = MTCNN(keep_all=True, device=DEVICE, thresholds=MTCNN_THRESHOLDS, post_process=True)

class FaceTracker:
    def __init__(self, max_disappeared=5):
        self.next_face_id = 0
        self.faces = {}
        self.scores = defaultdict(list)
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.faces[self.next_face_id] = centroid
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1
        return self.next_face_id - 1

    def update(self, rects):
        if len(rects) == 0:
            for fid in list(self.disappeared.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    self.deregister(fid)
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2)/2), int((y1 + y2)/2))

        if len(self.faces) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            face_ids = list(self.faces.keys())
            object_centroids = list(self.faces.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                fid = face_ids[row]
                self.faces[fid] = input_centroids[col]
                self.disappeared[fid] = 0
                used_rows.add(row)
                used_cols.add(col)

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col])

        return self.faces

    def deregister(self, face_id):
        del self.faces[face_id]
        del self.disappeared[face_id]

def analyze_frame_faces(frame, tracker):
    h, w, _ = frame.shape
    roi_y1, roi_y2 = int(h*0.10), int(h*0.85)
    rgb_frame = cv2.cvtColor(frame[roi_y1:roi_y2, :], cv2.COLOR_BGR2RGB)

    try:
        boxes, _ = mtcnn.detect(rgb_frame)
    except:
        return

    if boxes is None: return

    tracker.update(boxes)

    for box in boxes:
        centroid = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
        face_id = min(tracker.faces.keys(), key=lambda fid: np.linalg.norm(np.array(centroid) - np.array(tracker.faces[fid])))
        x1, y1, x2, y2 = box
        pad_w, pad_h = int((x2-x1)*0.25), int((y2-y1)*0.25)
        x1_p, y1_p = max(0,int(x1-pad_w)), max(0,int(y1-pad_h))
        x2_p, y2_p = min(rgb_frame.shape[1],int(x2+pad_w)), min(rgb_frame.shape[0],int(y2+pad_h))
        face_crop = rgb_frame[y1_p:y2_p, x1_p:x2_p]
        if face_crop.size==0: continue

        inputs = processor_face(images=face_crop, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model_face(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            tracker.scores[face_id].append(probs[0][FAKE_LABEL_INDEX].item())

def analyze_video_artifacts(frame):
    h, w, _ = frame.shape
    top_bar = np.mean(frame[:int(h*0.05), :, :])
    bottom_bar = np.mean(frame[int(h*0.95):, :, :])
    cinematic_bar_score = 1 if top_bar<CINEMATIC_BAR_THRESHOLD and bottom_bar<CINEMATIC_BAR_THRESHOLD else 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return cinematic_bar_score, blur_score

def check_video_authenticity(path, max_frames=MAX_FRAMES_TO_ANALYZE):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return {"verdict":"Error","message":"Could not open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, max_frames, dtype=int) if total_frames>max_frames else range(total_frames)

    tracker = FaceTracker()
    artifact_scores, blur_scores = [], []
    start_time = time.time()

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue

        analyze_frame_faces(frame, tracker)
        cinematic_score, blur_score = analyze_video_artifacts(frame)
        artifact_scores.append(cinematic_score)
        blur_scores.append(blur_score)

    cap.release()
    total_time = time.time()-start_time

    worst_face_score = max([np.mean(s) for s in tracker.scores.values()], default=0)
    worst_instability = max([np.std(s) for s in tracker.scores.values()], default=0)
    avg_cinematic = np.mean(artifact_scores)
    avg_blur = np.mean(blur_scores)

    verdict = "Real / Authentic"
    label = "SAFE: No significant manipulations detected."

    if worst_face_score>FACE_FAKE_THRESHOLD:
        verdict = "Deepfake / Manipulated"
        label = f"WARNING: Faces show strong manipulation (max score: {worst_face_score:.2f})"
    elif worst_instability>TEMPORAL_VARIANCE_THRESHOLD and avg_cinematic>0.5:
        verdict = "Cinematic / Movie Video"
        label = "INFO: Temporal jitter and cinematic bars detected; likely movie/trailer footage"
    elif worst_face_score>SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious / Possible Manipulation"
        label = f"CAUTION: Faces show moderate anomaly (score: {worst_face_score:.2f})"

    return {
        "verdict": verdict,
        "summary": label,
        "authenticity_score": round((1-worst_face_score)*100,2),
        "ai_probability": round(worst_face_score*100,2),
        "temporal_jitter": round(worst_instability,4),
        "faces_analyzed": len(tracker.scores),
        "cinematic_bar_score": round(avg_cinematic*100,2),
        "average_blur": round(avg_blur,2),
        "processing_time_sec": round(total_time,2),
        "metadata":{"model_face":MODEL_NAME_FACE,"frames_processed":len(indices)}
    }

if __name__=="__main__":
    video_path = "input_video.mp4"
    print("\n"+"="*60)
    print("      INDUSTRY-LEVEL DEEPFAKE FORENSIC REPORT")
    print("="*60)
    result = check_video_authenticity(video_path)
    for k,v in result.items():
        if k!="metadata": print(f"{k.upper():<25}: {v}")
    print("="*60+"\n")
