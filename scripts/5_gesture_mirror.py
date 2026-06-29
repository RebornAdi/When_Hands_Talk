"""
5_gesture_mirror.py
Bonus feature: shows your hand landmarks mirrored onto a separate
stylized canvas, side by side with the real camera feed. Purely visual,
no classification involved - a nice extra for a demo video.
"""

import cv2
from helpers import (
    initialize_hands, initialize_camera, create_mirror_canvas,
    draw_stylized_hand, flip_frame
)


def main():
    hands = initialize_hands()
    cap = initialize_camera()

    print("✋ Gesture Mirror running. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = flip_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        canvas = create_mirror_canvas()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_stylized_hand(canvas, hand_landmarks)

        cv2.imshow("Camera", frame)
        cv2.imshow("Gesture Mirror", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
