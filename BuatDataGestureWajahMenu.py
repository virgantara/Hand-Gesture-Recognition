import pygame
import sys
import numpy as np
import MdGesture as KL 
#import MdFiturWajah as FW 
import MdFiturTangan as FW 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model

def Model1(X,y) : 
    print(X.shape)
    
    
    #######################################
    #Bagian Ini Jangan Diubah
    #--------------------------------------
    nKelas = y.shape[1]
    s = X.shape 
    input_shape = X.shape[1:]  
    #######################################
    #Bagian ini adalah yang boleh diubah 
    #######################################
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(nKelas, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def Model(X,y) :     
    
    #######################################
    #Bagian Ini Jangan Diubah
    #--------------------------------------
    nKelas = y.shape[1]
    s = X.shape 
    input_shape = X.shape[1:]  
    model = Sequential()

    # Menambahkan TimeDistributed CNN layers
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layer untuk mengolah urutan fitur dari setiap frame
    model.add(LSTM(64, activation='relu'))
    
    # Output layer untuk klasifikasi
    model.add(Dense(nKelas, activation='softmax'))
    
    # Kompilasi model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

def run_menu(JenisGesture,NamaModel ="ModelTangan.h5",  Dataset="DataSetTangan", labels = ["Kiri","Kanan"]):
    
    global X,y
    Gesture = KL.CLGesture(JenisGesture)
    Gesture.Kelas = labels 
    Gesture.NamaModel = NamaModel
    Gesture.NoKamera = 0 
    Gesture.DirektoriDataSet ="DataSetTangan"
    
        
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
    X=[]
    y=[] 
    
    # Font settings
    font = pygame.font.Font(None, 36)
    
    # Menu options
    menu_options = []
    for i in range(len(labels)):
        menu_options.append(f"{i}. Buat Data Set : {labels[i]}") 
        
    nn =[ "",  
         "H. Hitung Data Set",  
         "W. Tes Webcam", 
         "L. Gesture Load Data Set",
         "T. Train Model",
         "K. Klasifikasi Gestur",
         "D. Delete Dataset Directory",
         "q. Quit"
    ]
    for i in range(len(nn)):
        menu_options.append(f"{nn[i]}") 
    
    
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
    NLabel = len(labels) 
    while running:
        screen.fill(white)

        # Draw menu options on screen
        for i, text_surface in enumerate(menu_texts):
            if i<NLabel : 
                screen.blit(text_surface, (screen_width // 10, 50 + i * 40))
            else :
                screen.blit(text_surface, (screen_width // 10*5, 50 + (i-NLabel-1) * 40))

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                
                if  event.key == pygame.K_h:
                    
                    Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_w:
                    Gesture.TesWebCam()
                elif event.key == pygame.K_0:
                    if NLabel>0 : 
                        Gesture.BuatKelasGesture(labels[0])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_1:
                    if NLabel>1 : 
                        Gesture.BuatKelasGesture(labels[1])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_2:
                    if NLabel>2 : 
                        Gesture.BuatKelasGesture(labels[2])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_3:
                    if NLabel>3 : 
                        Gesture.BuatKelasGesture(labels[3])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_4:
                    if NLabel>4 : 
                        Gesture.BuatKelasGesture(labels[4])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_5:
                    if NLabel>5 : 
                        Gesture.BuatKelasGesture(labels[5])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_6: 
                    if NLabel>6 : 
                        Gesture.BuatKelasGesture(labels[6])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_7: 
                    if NLabel>7 : 
                        Gesture.BuatKelasGesture(labels[7])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_8: 
                    if NLabel>8 : 
                        Gesture.BuatKelasGesture(labels[8])
                        Gesture.HitungDirektoriSetiapKelas(labels)
                elif event.key == pygame.K_9: 
                    if NLabel>9 : 
                        Gesture.BuatKelasGesture(labels[9])
                        Gesture.HitungDirektoriSetiapKelas(labels)
        
                elif event.key == pygame.K_t:
                    Gesture.TrainingGesture(Model)
                elif event.key == pygame.K_k:
                    Gesture.KlasifikasiGesture()
                elif event.key == pygame.K_d:
                    if confirmation_prompt():
                        Gesture.HapusDirektoriDataSet()
                        print("Dataset directory deleted.")
                    else:
                        print("Deletion canceled.")
                elif (event.key == pygame.K_q ):
                    pygame.quit()
                    sys.exit()
              

                
                    

# Run the menu
run_menu(
    FW.CLFiturGesture(),
    NamaModel ="ModelTangan.h5",  
    Dataset="DataSetTangan", labels = ["Kiri","Kanan"]
)




