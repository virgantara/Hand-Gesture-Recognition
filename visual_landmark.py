import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

# Initialize Mediapipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Default window size
cv2.namedWindow('Hand Landmark Detection', cv2.WINDOW_NORMAL)

is_maximized = False  # Maximize/minimize status

def ekstraksi_fitur(frame):
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = hands.process(frame_rgb)

    # Frame dimensions
    height, width, _ = frame.shape

    # Initialize numpy arrays for left and right hands
    left_hand_landmarks = np.zeros((21, 2))
    right_hand_landmarks = np.zeros((21, 2))

    # Helper function to process landmarks
    def process_landmarks(hand_landmarks, width, height):
        landmarks = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]
        landmark_0 = np.array(landmarks[0])
        landmark_5 = np.array(landmarks[5])
        normalized_landmarks = [
            ((x - landmark_0[0]) / (landmark_5[0] - landmark_0[0] + 1e-6),
             (y - landmark_0[1]) / (landmark_5[1] - landmark_0[1] + 1e-6))
            for x, y in landmarks
        ]
        return np.array(normalized_landmarks)

    # If hands are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Identify hand as left or right
            handedness = hand_handedness.classification[0].label
            processed_landmarks = process_landmarks(hand_landmarks, width, height)
            if handedness == 'Left':
                left_hand_landmarks = processed_landmarks
            elif handedness == 'Right':
                right_hand_landmarks = processed_landmarks

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Concatenate left and right hand landmarks
    concatenated_landmarks = np.concatenate((left_hand_landmarks.flatten(), right_hand_landmarks.flatten()))
    return concatenated_landmarks

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam")
        break

    # Process landmarks and draw them on the frame
    concatenated_landmarks = ekstraksi_fitur(frame)

    print("Concatenated Landmarks:", concatenated_landmarks)

    # Display the frame with landmarks
    cv2.imshow('Hand Landmark Detection', frame)

    # Handle key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit on 'q'
        break
    elif key == ord('m'):  # Maximize/minimize window
        if is_maximized:
            cv2.setWindowProperty('Hand Landmark Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            is_maximized = False
        else:
            cv2.setWindowProperty('Hand Landmark Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            is_maximized = True

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
