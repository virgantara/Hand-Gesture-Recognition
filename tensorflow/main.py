import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os
import time
from sklearn.model_selection import train_test_split 
from keras.optimizers import SGD
from model import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def feature_extract(frame, hands):
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

def read_landmarks(dataset, labels):
    # Initialize Mediapipe Hands and Drawing
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    kelas = np.eye(len(labels))
    y = []
    """
    Reads all `.jpg` images from multiple label directories inside the dataset.

    Args:
        dataset (str): The base directory of the dataset.
        labels (list): List of label names (subdirectories).

    Returns:
        dict: A dictionary where keys are label names and values are lists of image frames.
    """
    images_by_label = {}
    X = []
    for i, label in enumerate(labels):
        # Construct the directory path
        directory_path = os.path.join(dataset, label)

        # Check if the directory exists
        if not os.path.exists(directory_path):
            print(f"Directory '{directory_path}' does not exist.")
            images_by_label[label] = []
            continue

        # List all `.jpg` files in the directory
        filenames = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]

        # Read images and store them as frames
        # frames = []
        for filename in filenames:
            file_path = os.path.join(directory_path, filename)
            frame = cv2.imread(file_path)  # Read the image with OpenCV
            fitur = feature_extract(frame, hands)
            X.append(fitur)
            y.append(kelas[i])

    hands.close()
    return np.array(X), np.array(y)



# Prepare the dataset for PyTorch
def prepare_data(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


# Example usage
dataset = "../dataset"                  # Base dataset directory
labels = ["Satu", "Dua"]     # List of labels (subdirectories)

# Call the function
X,y = read_landmarks(dataset, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Parameters
input_size = X_train.shape[1]  # Number of features (flattened landmarks)
num_classes = len(labels)      # Number of labels
batch_size = 32                # Batch size for DataLoader
learning_rate = 0.001          # Learning rate
num_epochs = 50                # Number of epochs

# Create model, loss function, and optimizer
model = get_model(input_size=input_size, num_classes=num_classes)

# Compile the model
model.compile(optimizer=SGD(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f} %")

# Make predictions (optional)
predictions = model.predict(X_test)
predicted_labels = tf.argmax(predictions, axis=1).numpy()
# print(predicted_labels)
