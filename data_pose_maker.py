import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os
import time

def create_dataset(direktori_path, nama_label, frameratesave=2, maxframe=100, start_delay=5):
    """
    Capture webcam frames, process, and save them to a dataset with a specified label.
    
    Args:
        direktori_path (str): Path to save the dataset.
        nama_label (str): Label for the dataset.
        frameratesave (int): Frame saving rate (frames per second).
        maxframe (int): Maximum number of frames to save.
        start_delay (int): Delay in seconds before saving starts.
    """
    # Initialize Mediapipe Hands and Drawing
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    # Create directory path
    save_dir = os.path.join(direktori_path, nama_label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")

    # Default window size
    cv2.namedWindow('Hand Landmark Detection', cv2.WINDOW_NORMAL)

    # Initialize variables
    time_before = datetime.now()
    start_time = time.time()
    counter = 0

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

        # Countdown before saving starts
        elapsed_time = time.time() - start_time
        if elapsed_time < start_delay:
            countdown_text = f"Starting in {int(start_delay - elapsed_time)} seconds"
            cv2.putText(frame, countdown_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Hand Landmark Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Check framerate and save frame
        time_now = datetime.now()
        if (time_now - time_before).total_seconds() > 1 / frameratesave:
            # Generate file name
            file_name = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + ".jpg"
            file_path = os.path.join(save_dir, file_name)

            # Save frame
            cv2.imwrite(file_path, frame)
            counter += 1

            # Update the previous time
            time_before = time_now

        # Display counter on frame
        cv2.putText(frame, f"Files saved: {counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process landmarks and draw them on the frame
        concatenated_landmarks = ekstraksi_fitur(frame)
        print("Concatenated Landmarks:", concatenated_landmarks)

        # Display the frame with landmarks
        cv2.imshow('Hand Landmark Detection', frame)

        # Stop saving after maxframe is reached
        if counter >= maxframe:
            print(f"Reached maximum of {maxframe} frames. Exiting...")
            break

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Exit on 'q'
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    create_dataset(direktori_path="dataset", nama_label="Dua", frameratesave=2, maxframe=50, start_delay=3)