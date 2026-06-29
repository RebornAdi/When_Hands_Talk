
import os
import time
import csv
import argparse

import cv2
import mediapipe as mp
import numpy as np

from config import (
    GESTURE_LABELS, NUM_GESTURES, SAMPLES_PER_GESTURE,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def initialize_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


def save_metadata(gesture_dir, rows):
    with open(os.path.join(gesture_dir, "meta.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "timestamp"])
        writer.writerows(rows)


def existing_sample_count(gesture_dir):
    if not os.path.isdir(gesture_dir):
        return 0
    return len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])


def collect_for_gesture(cap, hands, mp_draw, gesture_idx, label):
    gesture_dir = os.path.join(DATA_DIR, f"{gesture_idx}_{label}")
    os.makedirs(gesture_dir, exist_ok=True)

    print(f"\nPrepare for Gesture {gesture_idx} ({label})")
    print("Press 's' to start capturing, 'n' to skip this gesture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"Press 's' to capture '{label}'  |  'n' to skip",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            break
        if key == ord('n'):
            print(f"Skipped gesture '{label}'.")
            return

    for sec in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"Starting in {sec}",
                    (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Capture", frame)
        cv2.waitKey(1000)

    metadata = []
    count = 0
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        display = frame.copy()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand, mp.solutions.hands.HAND_CONNECTIONS)

            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            filename = f"{count}.npy"
            np.save(os.path.join(gesture_dir, filename), np.array(data, dtype=np.float32))
            metadata.append([filename, int(time.time())])
            count += 1

        cv2.putText(display, f"Gesture {label}: {count}/{SAMPLES_PER_GESTURE}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture", display)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Aborted early by user - keeping samples captured so far.")
            break

    save_metadata(gesture_dir, metadata)
    print(f"✅ Collected {count} samples for '{label}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                         help="Collect samples for only this single label, e.g. --only M")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip gestures that already have enough samples (for resuming)")
    args = parser.parse_args()

    hands = initialize_hands()
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    for gesture_idx in range(NUM_GESTURES):
        label = GESTURE_LABELS[gesture_idx]

        if args.only is not None and label != args.only:
            continue

        gesture_dir = os.path.join(DATA_DIR, f"{gesture_idx}_{label}")
        if args.skip_existing and existing_sample_count(gesture_dir) >= SAMPLES_PER_GESTURE:
            print(f"Skipping '{label}' - already has enough samples.")
            continue

        collect_for_gesture(cap, hands, mp_draw, gesture_idx, label)

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Data collection complete.")


if __name__ == "__main__":
    main()
