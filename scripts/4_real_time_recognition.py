import cv2
import mediapipe as mp
import joblib
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from config import (
    MODEL_DIR, MODEL_NAME, SCALER_NAME,
    GESTURE_LABELS, SMOOTHING_ALPHA, VOTE_WINDOW,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS
)

# -------------------------------
# Load scaler
# -------------------------------
scaler = joblib.load(f"{MODEL_DIR}/{SCALER_NAME}")

# -------------------------------
# SAME MODEL ARCHITECTURE AS TRAINING
# -------------------------------
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

# -------------------------------
# Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.mean_.shape[0]

model = MLP(input_dim, len(GESTURE_LABELS))
model.load_state_dict(torch.load(f"{MODEL_DIR}/{MODEL_NAME}", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Setup Mediapipe
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Smoothing buffers
# -------------------------------
pred_buffer = deque(maxlen=VOTE_WINDOW)
smooth_landmarks = None

cap = cv2.VideoCapture(0)

print("🎉 Real-time ASL Recognition Started!")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    display = frame.copy()

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

        # Flatten 21×3 → 63
        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

        # Smooth landmarks using EWMA
        if smooth_landmarks is None:
            smooth_landmarks = raw
        else:
            smooth_landmarks = (
                SMOOTHING_ALPHA * raw +
                (1 - SMOOTHING_ALPHA) * smooth_landmarks
            )

        # Scale input
        X = scaler.transform([smooth_landmarks])[0]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(X_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        # Majority vote smoothing
        pred_buffer.append(pred)
        final_pred = max(set(pred_buffer), key=pred_buffer.count)
        label = GESTURE_LABELS[final_pred]

        # Display
        cv2.putText(display, f"{label} ({probs[pred]:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    cv2.imshow("ASL A-D Recognition", display)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
