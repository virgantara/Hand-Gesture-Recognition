import pygame
import sys

import numpy as np

import MdGesture as KL 
import MdFiturWajah as FW 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, TimeDistributed, LeakyReLU

from tensorflow.keras.layers import Dropout
   


def Training(X,y,Epoh=400) :
    nKelas = y.shape[1] 
    model = Sequential()
    sp=(X.shape[1], X.shape[2], X.shape[3])
    # First Conv1D layer with LeakyReLU activation
    model.add(TimeDistributed(Conv1D(filters=68, kernel_size=3), 
                              input_shape=sp))
    model.add(TimeDistributed(LeakyReLU(alpha=0.5)))

    # Second Conv1D layer with LeakyReLU activation
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3)))
    model.add(TimeDistributed(LeakyReLU(alpha=0.5)))

    # Flatten layer
    model.add(TimeDistributed(Flatten()))

    # LSTM layer with tanh activation (default)
    model.add(LSTM(units=128))

    # Output layer with softmax for multi-class classification
    model.add(Dense(1000, activation='linear'))
    model.add(Dropout(0.5)) 
    model.add(Dense(1000, activation='linear'))
    model.add(Dropout(0.5)) 
    model.add(Dense(nKelas, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Pelatihan model
    model.fit(X, y, epochs=Epoh, batch_size=1, validation_split=0.2,shuffle=True)
    # Evaluasi model
    loss, accuracy = model.evaluate(X, y)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return model
   
labels = ["Kanan", "Kiri","Lurus","Maju","Mundur"]        
Gesture = KL.CLGesture(FW.CLFiturFaceGesture())




def run_menu():
        
    # Initialize Pygame
    pygame.init()
    
    # Screen settings
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Gesture Capture and Classification Menu")
    
    # Colors
    white = (255, 255, 255)

    blue = (0, 0, 255)
    red = (255, 0, 0)
    
    
    # Font settings
    font = pygame.font.Font(None, 36)
    
    # Menu options
    menu_options = [
        "1. Capture Gesture: Maju",
        "2. Capture Gesture: Kiri",
        "3. Capture Gesture: Kanan",
        "4. Capture Gesture: Lurus",
        "5. Capture Gesture: Mundur",
        "6. Train Model",
        "7. Klasifikasi Gestyre",
        "8. Delete Dataset Directory",
        "9. Quit"
    ]
    
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
            screen.blit(text_surface, (screen_width // 4, 100 + i * 50))

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    Gesture.BuatKelasGesture("Maju")
                elif event.key == pygame.K_2:
                    Gesture.BuatKelasGesture("Kiri")
                elif event.key == pygame.K_3:
                    Gesture.BuatKelasGesture("Kanan")
                elif event.key == pygame.K_4:
                    Gesture.BuatKelasGesture("Lurus")
                elif event.key == pygame.K_5:
                    Gesture.BuatKelasGesture("Mundur")
                    
                elif event.key == pygame.K_6:
                    Gesture.TrainingGesture(Training,labels)
                elif event.key == pygame.K_7:
                    Gesture.KlasifikasiGesture(labels)
                elif event.key == pygame.K_8:
                    if confirmation_prompt():
                        Gesture.HapusDirektoriDataSet()
                        print("Dataset directory deleted.")
                    else:
                        print("Deletion canceled.")
                elif event.key == pygame.K_9 or event.key == pygame.K_q or event.key == pygame.K_Q:
                    pygame.quit()
                    sys.exit()

# Run the menu
run_menu()
#X,y = Gesture.LoadDataSet(labels)


