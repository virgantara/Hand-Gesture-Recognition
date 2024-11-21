import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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