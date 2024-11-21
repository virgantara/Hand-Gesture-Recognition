import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os
import time
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Initialize Mediapipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

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

def read_all_images(dataset, labels):

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
            fitur = ekstraksi_fitur(frame)
            X.append(fitur)
            y.append(kelas[i])
        # images_by_label[label] = frames

    return np.array(X), np.array(y)


# Define the CNN Model
class HandGestureCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandGestureCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Fully connected layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Prepare the dataset for PyTorch
def prepare_data(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


# Example usage
dataset = "dataset"                  # Base dataset directory
labels = ["Satu", "Dua"]     # List of labels (subdirectories)

# Call the function
X,y = read_all_images(dataset, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)




hands.close()

# Parameters
input_size = X_train.shape[1]  # Number of features (flattened landmarks)
num_classes = len(labels)      # Number of labels
batch_size = 32                # Batch size for DataLoader
learning_rate = 0.001          # Learning rate
num_epochs = 20                # Number of epochs

# Create model, loss function, and optimizer
model = HandGestureCNN(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare data loaders
train_dataset = prepare_data(X_train, y_train)
test_dataset = prepare_data(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))  # One-hot to index
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Testing loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")