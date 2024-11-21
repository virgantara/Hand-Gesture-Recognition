# Hand Gesture Recognition with Mediapipe and PyTorch

This repository implements a hand gesture recognition system using Python, Mediapipe, and PyTorch. The system detects and classifies hand gestures in real-time, making it suitable for applications like sign language recognition, gesture-based control systems, and interactive applications.

---

## Features

- Real-time hand gesture detection using Mediapipe.
- Gesture classification with a PyTorch-based neural network.
- Easy-to-extend codebase for adding more gestures or fine-tuning models.
- Supports live video input and pre-recorded video files.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/virgantara/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
```

## Dataset Creation
The code is intended to capture hand pose for the feature extraction purpose.
Run the `data_pose_maker.py` with:
```bash
python data_pose_maker.py
```

## Tensorflow module
1. Change to tensorflow directory with `cd tensorflow`
2. Run the `main.py` file to train and evaluate:
```bash
python main.py
```

## PyTorch module
1. Change to pytorch directory with `cd pytorch`
2. Run the `main.py` file to train and evaluate:
```bash
python main.py
```