"""
4_real_time_recognition.py
Local desktop demo: webcam -> landmark extraction -> MLP classification
-> sentence assembly -> on-screen text -> optional speech output.

Keys while running:
    ESC - quit
    v   - speak the current sentence aloud
    c   - clear the current sentence
"""

import json
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from config import (
    MODEL_DIR, MODEL_NAME, SCALER_NAME, LABEL_MAP_NAME,
    SMOOTHING_ALPHA, VOTE_WINDOW,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS,
    STABLE_FRAMES_REQUIRED, COMMIT_COOLDOWN_SECONDS,
    SPACE_LABEL, DELETE_LABEL, NO_HAND_LABEL
)
from sentence_builder import SentenceBuilder
from tts_engine import speak_offline


# Load the label map saved at training time - keeps inference in sync
# with whatever classes the model was actually trained on, even if
# config.py has since changed.
with open(os.path.join(MODEL_DIR, LABEL_MAP_NAME)) as f:
    raw_map = json.load(f)
LABEL_MAP = {int(k): v for k, v in raw_map.items()}

scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_NAME))


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.mean_.shape[0]
model = MLP(input_dim, len(LABEL_MAP))
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=device))
model.to(device)
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

pred_buffer = deque(maxlen=VOTE_WINDOW)
smooth_landmarks = None
builder = SentenceBuilder(
    stable_frames_required=STABLE_FRAMES_REQUIRED,
    cooldown_seconds=COMMIT_COOLDOWN_SECONDS,
    space_label=SPACE_LABEL,
    delete_label=DELETE_LABEL,
    no_hand_label=NO_HAND_LABEL,
)

cap = cv2.VideoCapture(0)
print("🎉 Real-time recognition started. ESC to quit, 'v' to speak, 'c' to clear.")

frame_times = deque(maxlen=30)  # rolling window for FPS measurement

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    display = frame.copy()

    label = NO_HAND_LABEL
    conf = 0.0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

        if smooth_landmarks is None:
            smooth_landmarks = raw
        else:
            smooth_landmarks = SMOOTHING_ALPHA * raw + (1 - SMOOTHING_ALPHA) * smooth_landmarks

        X = scaler.transform([smooth_landmarks])[0]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(X_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        pred_buffer.append(pred)
        final_pred = max(set(pred_buffer), key=pred_buffer.count)
        label = LABEL_MAP[final_pred]
        conf = float(probs[pred])
    else:
        smooth_landmarks = None
        pred_buffer.clear()

    builder.update(label)

    frame_times.append(time.time() - t0)
    fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    cv2.putText(display, f"{label} ({conf:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"Sentence: {builder.get_sentence()}", (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Sign Recognition", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('v'):
        text = builder.get_sentence().strip()
        if text:
            speak_offline(text)
    elif key == ord('c'):
        builder.clear()

cap.release()
cv2.destroyAllWindows()
