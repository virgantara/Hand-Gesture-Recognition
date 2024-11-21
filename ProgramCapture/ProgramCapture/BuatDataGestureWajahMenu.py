import pygame
import sys
import numpy as np
import MdGesture as KL 
import MdFiturWajah as FW 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_training_history_and_save_csv(history, model, X_test, y_test, class_names):
    # Generate directory name with current timestamp (YYMMDDHHmmssms format)
    dir1 = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
    save_dir = os.path.join("Report", dir1)
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract history data
    history_data = {
        'epoch': list(range(1, len(history.history['accuracy']) + 1)),
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy_difference': [a - v for a, v in zip(history.history['accuracy'], history.history['val_accuracy'])],
        'loss_difference': [l - v for l, v in zip(history.history['loss'], history.history['val_loss'])]
    }
    
    # Save history data to CSV
    history_df = pd.DataFrame(history_data)
    history_csv_path = os.path.join(save_dir, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    
    # Plot training and validation accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    accuracy_path = os.path.join(save_dir, 'model_accuracy.png')
    plt.savefig(accuracy_path)
    plt.show()  # Display the plot

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    loss_path = os.path.join(save_dir, 'model_loss.png')
    plt.savefig(loss_path)
    plt.show()  # Display the plot

    # Plot accuracy difference
    plt.figure(figsize=(8, 4))
    plt.plot(history_data['accuracy_difference'], label='Accuracy Difference (Train - Validation)')
    plt.title('Difference Between Train and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference')
    plt.legend(loc='upper right')
    accuracy_diff_path = os.path.join(save_dir, 'accuracy_difference.png')
    plt.savefig(accuracy_diff_path)
    plt.show()  # Display the plot

    # Plot loss difference
    plt.figure(figsize=(8, 4))
    plt.plot(history_data['loss_difference'], label='Loss Difference (Train - Validation)')
    plt.title('Difference Between Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend(loc='upper right')
    loss_diff_path = os.path.join(save_dir, 'loss_difference.png')
    plt.savefig(loss_diff_path)
    plt.show()  # Display the plot

    # Confusion Matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)  # Predictions
    y_true = np.argmax(y_test, axis=1)                 # True labels
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix data to CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(save_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.show()  # Display the plot

    print(f"Plots and CSV files saved in directory: {save_dir}")
    

def Training(X, y, Epoh, Kelas):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)

    # Define the number of classes and input shape
    nKelas = y.shape[1]
    input_shape = (X.shape[1], X.shape[2])
    # Initialize the model
    # Menambahkan LSTM tambahan dan Dense tambahan
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(nKelas, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Display model summary
    model.summary()
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Train the model
    history = model.fit(X_train, y_train, epochs=Epoh, batch_size=32, validation_data=(X_test, y_test),
                        shuffle=True, callbacks=[early_stopping])
    # Plot training history and confusion matrix
    plot_training_history_and_save_csv(history, model, X_test, y_test, Kelas)
    return model


def run_menu(JenisGesture, Dataset="DataSet", labels = ["Kiri","Kanan","Maju","Lurus","Stop","nan"]):
    
    global X,y
    
    Gesture = KL.CLGesture(JenisGesture)

        
    # Initialize Pygame
    pygame.init()
    
    # Screen settings
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Gesture Capture and Classification Menu")
    
    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    X=[]
    y=[] 
    
    # Font settings
    font = pygame.font.Font(None, 36)
    
    # Menu options
    menu_options = [
        "a. Hitung Data Set",
        "w. Tes Webcam",
        "0. Gesture Load Data Set",
        "1. Capture Gesture: Kiri",
        "2. Capture Gesture: Kanan",
        "3. Capture Gesture: Maju",
     #   "4. Capture Gesture: Mundur",
        "5. Capture Gesture: Lurus",
        "6. Capture Gesture: Stop",
        "7. Capture Gesture: Nan",
        "T. Train Model",
        "K. Klasifikasi Gestur",
        "D. Delete Dataset Directory",
        "q. Quit"
    ]
    Gesture.HitungDirektoriSetiapKelas(labels)
    
    # Render text for each menu option
    menu_texts = [font.render(option, True, blue) for option in menu_options]
    
    def confirmation_prompt():
        screen.fill(white)
        prompt_text = font.render("Are you sure? (Y/N)", True, red)
        screen.blit(prompt_text, (screen_width // 3, screen_height // 2))
        pygame.display.flip()
    
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return True
                    elif event.key == pygame.K_n:
                        return False
    

    running = True
    while running:
        screen.fill(white)

        # Draw menu options on screen
        for i, text_surface in enumerate(menu_texts):
            screen.blit(text_surface, (screen_width // 4, 50 + i * 40))

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_0 ):
                    X,y = Gesture.LoadDataSet(labels)
                elif event.key == pygame.K_a:
                    
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_w:
                    
                    Gesture.TesWebCam()
                
                
        
                elif event.key == pygame.K_1:
                    Gesture.BuatKelasGesture("Kiri")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_2:
                    Gesture.BuatKelasGesture("Kanan")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_3:
                    Gesture.BuatKelasGesture("Maju")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_4:
                    Gesture.BuatKelasGesture("Mundur")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_5:
                    Gesture.BuatKelasGesture("Lurus")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_6:
                    Gesture.BuatKelasGesture("Stop")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_7: 
                    Gesture.BuatKelasGesture("Nan")
                    Gesture.HitungDirektoriSetiapKelas(labels)
                
                    
                    
                    
                    
                elif event.key == pygame.K_t:
                    Gesture.TrainingGesture(Training,labels,100)
                elif event.key == pygame.K_k:
                    Gesture.KlasifikasiGesture(labels)
                elif event.key == pygame.K_d:
                    if confirmation_prompt():
                        Gesture.HapusDirektoriDataSet()
                        print("Dataset directory deleted.")
                    else:
                        print("Deletion canceled.")
                elif (event.key == pygame.K_q )or (event.key == pygame.K_Q):
                    pygame.quit()
                    sys.exit()
              

                
                    

# Run the menu
run_menu(FW.CLFiturFaceGesture())




