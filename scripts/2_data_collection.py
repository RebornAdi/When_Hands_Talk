import os, time, csv
import cv2, mediapipe as mp
import numpy as np
from config import NUM_GESTURES, SAMPLES_PER_GESTURE, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def initialize_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

def save_metadata(gesture_dir, rows):
    with open(os.path.join(gesture_dir, "meta.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "timestamp"])
        writer.writerows(rows)

def main():
    hands = initialize_hands()
    cap = cv2.VideoCapture(0)
    mp_draw = mp.solutions.drawing_utils

    for gesture_idx in range(NUM_GESTURES):

        print(f"\nPrepare for Gesture {gesture_idx} ({gesture_idx})")
        print("Press 's' to start capturing samples...")

        # prep window
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f"Press 's' to capture Gesture {gesture_idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Capture", frame)
            if cv2.waitKey(25) & 0xFF == ord('s'):
                break

        # countdown
        for sec in range(3,0,-1):
            ret, frame = cap.read()
            cv2.putText(frame, f"Starting in {sec}",
                        (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            cv2.imshow("Capture", frame)
            cv2.waitKey(1000)

        # actual collection
        gesture_dir = os.path.join(DATA_DIR, f"{gesture_idx}_{['A','B','C','D'][gesture_idx]}")
        os.makedirs(gesture_dir, exist_ok=True)
        metadata = []
        count = 0

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            display = frame.copy()

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display, hand, mp.solutions.hands.HAND_CONNECTIONS)

                # flatten 21×3 → 63
                data = []
                for lm in hand.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                filename = f"{count}.npy"
                np.save(os.path.join(gesture_dir, filename), np.array(data))
                metadata.append([filename, int(time.time())])
                count += 1
                print(f"Collected: {count}/{SAMPLES_PER_GESTURE}")

            cv2.putText(display, f"Gesture {gesture_idx}: {count}/{SAMPLES_PER_GESTURE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Capture", display)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        save_metadata(gesture_dir, metadata)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
