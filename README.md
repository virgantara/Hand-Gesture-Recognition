# Hand Gesture Recognition with Mediapipe and PyTorch

This repository implements a hand gesture recognition system using Python, Mediapipe, Tensorflow, and PyTorch. The system detects and classifies hand gestures in real-time, making it suitable for applications like sign language recognition, gesture-based control systems, and interactive applications. You can select any modules you prefer, such as Tensorflow or PyTorch.

---

## Features

- Real-time hand gesture detection using Mediapipe.
- Gesture classification with a PyTorch-based neural network.
- Easy-to-extend codebase for adding more gestures or fine-tuning models.
- Supports live video input and pre-recorded video files.

---

## Installation
### Conda Installer Path
You can download the conda installer path in [here](https://www.anaconda.com/download/success)
### Conda Environment Installation
Run the following code:
```bash
conda create --name hand-gesture-env python=3.8
```

### Conda Environment Activation
```bash
conda activate hand-gesture-env
```

1. Clone the repository:
```bash
git clone https://github.com/virgantara/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
```
1. Python required libraries installation
```bash
pip install -r requirements.txt
```


## Dataset Creation
The code is intended to capture hand pose for the feature extraction purpose.
Run the `data_pose_maker.py` with:
```bash
python data_pose_maker.py --label_name=Three
```
### List of available params
1. `--label_name` : Name of the label (default: One)
1. `--dataset_path` : The path of dataset (default: dataset)
1. `--frameratesave` : Frame Rate checkpoint for saving image (default: 2)
1. `--maxframe` : Max Frame (default: 50)
1. `--start_delay` : Countdown time to delay before capturing (default: 3 seconds)

## Tensorflow module
### Run by terminal
1. Change to tensorflow directory with `cd tensorflow`
2. Run the `main.py` file to train and evaluate:
```bash
python main.py
```
### Run by jupyter notebook
1. Change to tensorflow directory with `cd tensorflow`
1. Start your jupyter notebook by running this command
```bash
jupyter notebook
```
1. Click the hand-landmark-detection.ipynb file


## PyTorch module
### Run by terminal
1. Change to pytorch directory with `cd pytorch`
2. Run the `main.py` file to train and evaluate:
```bash
python main.py
```

### Run by jupyter notebook
1. Change to tensorflow directory with `cd pytorch`
1. Start your jupyter notebook by running this command
```bash
jupyter notebook
```
1. Click the hand-landmark-detection.ipynb file

# Please Cite here if you think this helpfull

`E. M. Yuniarno, “Sign Language Recognition Based on Geometric Features Using Deep Learning”, j. nas. pendidik. teknik. inform., vol. 13, no. 2, pp. 338–348, Jul. 2024.`