import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# MODEL
# =========================
class EmotionLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        return self.fc(x)

# =========================
# LOAD MODELS ONCE
# =========================
face_detector = YOLO("yolov8n-face-lindevs.pt")

emotion_model = EmotionLSTMModel()
emotion_model.load_state_dict(torch.load("FER_dinamic_LSTM_SAVEE.pt", map_location="cpu"))
emotion_model.eval()

feature_mapper = nn.Linear(2304, 512)

# =========================
# API ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # READ IMAGE FROM WEBSITE
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # FACE DETECTION
    results = face_detector(image)
    largest_face = None
    max_area = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_face = (x1, y1, x2, y2)

    if largest_face is None:
        return {"error": "no face detected"}

    x1, y1, x2, y2 = largest_face
    face = image[y1:y2, x1:x2]

    # PREPROCESS
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = torch.tensor(face).float() / 255.0
    face = face.view(1, -1)

    mapped = feature_mapper(face).unsqueeze(1)

    with torch.no_grad():
        out = emotion_model(mapped)
        emotion = EMOTION_LABELS[out.argmax(1).item()]

    return {"emotion": emotion}
