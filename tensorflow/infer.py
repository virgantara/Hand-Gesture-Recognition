import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from datetime import datetime

# Load the trained TensorFlow model
model = tf.keras.models.load_model("outputs/model.keras")

# Initialize Mediapipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Function to extract hand landmarks
def ekstraksi_fitur(frame, hands):
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

# Open webcam
cap = cv2.VideoCapture(0)

labels = ["One", "Two", "Three"]

# Main loop for inference


Kelas = []
SaveRate = 15
WaktuJeda = 4
WaktuRekamGesture = 4
DirektoriDataSet = DataSet
JumlahFrame = 20
NamaModel ="model_lstm.h5"
NoKamera = 0 
LebarWindow=20 
WIndowStep = 4

BefTime= datetime.now()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam")
        break

    frame = cv2.flip(frame, 1) 
    FrameSimpan = frame.copy() 

    

    current_time = datetime.now()

    if (current_time - BefTime).total_seconds() > 1 / SaveRate: 
        # Extract features using Mediapipe
        features = ekstraksi_fitur(frame, hands)

        # Check if valid features were detected (non-zero)
        if np.any(features):
            # Reshape for model input
            features = features.reshape(1, -1)

            # Perform inference
            predictions = model.predict(features)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = labels[predicted_index]

            # Display the prediction on the frame
            label_text = f"Prediction: {predicted_label}"
            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Landmark Detection and Inference", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
