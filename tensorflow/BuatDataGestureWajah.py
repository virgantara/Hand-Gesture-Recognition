
import cv2
import sys
import shutil
import numpy as np
import os
from datetime import datetime
import mediapipe as mp
import  math 
import tensorflow as tf

class CLFiturFaceGesture:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Inisialisasi deteksi wajah
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def ProsesFitur(self,RawFit,RawFitBef):
        
        if len(RawFitBef)>0 :
     
            Fitur = np.array(RawFit- RawFitBef)
           #q Fitur = np.concatenate(  (Fit,  RawFitBef))
        else :
            Fitur =[]         
        return Fitur
    
    def EkstraksiFitur(self, frame, NamaFile=None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        h, w, _ = frame.shape    
        Fitur = np.zeros((936,))
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            
            h, w, _ = frame.shape
            # Mendapatkan koordinat hidung dan kedua mata
            nose_tip = face_landmarks.landmark[1]
            right_eye = face_landmarks.landmark[33]
            left_eye = face_landmarks.landmark[263]
    
            # Konversi koordinat relatif ke piksel
            
            nose_tip_coords = (int(nose_tip.x *  w ), int(nose_tip.y * h))
            right_eye_coords = (int(right_eye.x *  w), int(right_eye.y * h))
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
    
            # Menghitung titik tengah antara kedua mata
            middle_eye_x = (right_eye_coords[0] + left_eye_coords[0]) // 2*w
            middle_eye_y = (right_eye_coords[1] + left_eye_coords[1]) // 2*h
            middle_eye_coords = (middle_eye_x, middle_eye_y)
    
            # Hitung jarak Euclidean antara hidung dan titik tengah mata
            distance = math.sqrt((nose_tip_coords[0] - middle_eye_coords[0])**2 + 
                                 (nose_tip_coords[1] - middle_eye_coords[1])**2)
            if distance<1: 
                distance = 1 

            # Ambil koordinat bounding box berdasarkan landmark wajah
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]
            x_coords  =np.float32(np.array(x_coords ))*  w 
            y_coords  =np.float32(np.array(y_coords ))*  h 
            

            
            
            x_coords =(x_coords -nose_tip.x) /distance
            y_coords =(y_coords -nose_tip.y) /distance
            
            Fitur  =np.float32( np.concatenate((x_coords  , y_coords ))) 
        return Fitur 
        
    def Close(self):
        self.face_mesh.close()

    def Capture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
        return frame.copy()

# Kode untuk menangkap wajah
class CLGesture:
    def __init__(self, Proses):
        self.Proses = Proses
        self.Kelas = []
        self.SaveRate = 15
        self.WaktuJeda = 4
        self.WaktuRekamGesture = 4
        self.DirektoriDataSet = "DataSet"
        self.JumlahFrame = 40
        self.NamaModel ="model_lstm.h5"
        self.NoKamera = 0 
        self.LebarWindow=30 
        self.WIndowStep = 4
    def HapusDirektoriDataSet(self):
        # Path ke direktori yang ingin dihapus
        dir_path = self.DirektoriDataSet
        def remove_readonly(func, path, _):
            """Mengubah izin untuk memungkinkan penghapusan jika dibatasi."""
            os.chmod(path, 0o777)
            func(path)

        try:
            # Menghapus direktori dengan pengaturan ulang izin jika diperlukan
            shutil.rmtree(dir_path, onerror=remove_readonly)
            print(f"Direktori '{dir_path}' dan semua isinya telah dihapus.")
        except Exception as e:
            print(f"Gagal menghapus direktori: {e}")

        
        
    
    def __del__(self):
        print("Exit")
        self.Proses.Close()
    
    def wr(self,array, window_size, step_size):
        nRows = array.shape[0]
        
        # Hitung jumlah windows berdasarkan ukuran window dan step
        num_windows = (nRows - window_size) // step_size + 1
    
        # Buat array kosong untuk menyimpan windows
        windows = np.array([array[i:i + window_size, ...] for i in range(0, num_windows * step_size, step_size)])
        
        return windows


    
    
    def BuatKelasGesture(self ,NamaKelas,SaveRate =None,JumlahFrame = None ,WaktuJeda= None):  
        Proses = self.Proses
        
        if SaveRate == None :
            SR =self.SaveRate 
        else :
            SR = SaveRate
        if WaktuJeda==None : 
            N =self.WaktuJeda  
        else: 
            N= WaktuJeda 
        if JumlahFrame ==None : 
            JumlahFrame =self.JumlahFrame 
        
        if not os.path.exists(self.DirektoriDataSet):
            os.makedirs(self.DirektoriDataSet)
        DIR1= os.path.join(self.DirektoriDataSet, NamaKelas)
        # Membuat direktori jika belum ada
        if not os.path.exists(DIR1):
            os.makedirs(DIR1)
        current_time = datetime.now()
        Dir2 = ""     
        # Membuka webcam (index 0 untuk webcam default)
        cap = cv2.VideoCapture(self.NoKamera)
        # Mengecek apakah webcam berhasil dibuka
        if not cap.isOpened():
            print("Tidak dapat mengakses webcam")
            sys.exit()
        
        # Menghitung waktu awal
        start_time = datetime.now()
        file_count = 0  # Menghitung jumlah file yang disimpan
        saving_started = False  # Menandai apakah penyimpanan frame telah dimulai
        
        # Loop utama
        while True:
            # Membaca frame dari webcam
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame")
                break
        
            frame = cv2.flip(frame, 1) 
            FrameSimpan = frame.copy() 
            if Proses : 
                frame = Proses.Capture(frame)
                
            # Mendapatkan waktu saat ini
            current_time = datetime.now()
        
            # Menampilkan counter sebelum penyimpanan dimulai
            if not saving_started:
                # Menghitung waktu berlalu sebelum penyimpanan dimulai
                elapsed_time = (current_time - start_time).total_seconds()
                # Menampilkan penghitung waktu pada frame sebelum mulai menyimpan
                cv2.putText(frame, f"Starting in: {int(N - elapsed_time)}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Jika waktu jeda telah tercapai, mulai penyimpanan
                if elapsed_time >= N:
                    saving_started = True
                    # Reset waktu untuk mulai penghitungan elapsed time dari awal penyimpanan
                    start_capture_time = current_time
                    BefTime = current_time  # Set waktu awal penyimpanan
                #end if             
            else: 
            
                # Menghitung waktu berlalu sejak penyimpanan dimulai
                elapsed_capture_time = (current_time - start_capture_time).total_seconds()
            
                # Menampilkan jumlah file yang disimpan pada frame setelah penyimpanan dimulai
                cv2.putText(frame, f"Files saved: {file_count}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Elapsed Time: {int(elapsed_capture_time)}s", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
                # Mengecek apakah waktunya untuk menyimpan frame
                if (current_time - BefTime).total_seconds() > 1 / SR:
                    timestamp = current_time.strftime('%y%m%d%H%M%S%f')[:-3]  # Format YYMMDDHHmmSSMS
                 
                    if file_count==0:
                        Dir2 = os.path.join(DIR1, f"{timestamp}")
                        if not os.path.exists(Dir2):
                            os.makedirs(Dir2)
                                        
                    filename = os.path.join(Dir2, f"{timestamp}.jpg")
                    cv2.imwrite(filename, FrameSimpan)
                    print(f"Frame disimpan: {filename}")
            
                    # Menambah jumlah file yang disimpan
                    file_count += 1
            
                    # Memperbarui waktu sebelumnya
                    BefTime = current_time
            
                # Mengecek apakah waktu penyimpanan telah tercapai
                if file_count >= JumlahFrame+1:
                    print("Waktu penyimpanan telah tercapai, reset...")
                    saving_started = False
                    start_time = current_time
                    file_count = 0
                # Menampilkan frame di jendela 'Webcam'
            cv2.imshow(NamaKelas, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #end if 
        #end while 
            # Cek apakah direktori ada
        if file_count<JumlahFrame :
            if os.path.exists(Dir2):
                # Hapus direktori beserta isinya
                if 0< file_count <JumlahFrame :  
                    shutil.rmtree(Dir2)
                    print(f"Direktori '{Dir2}' dan seluruh isinya telah dihapus.")
                else: 
                    print("Direktori Tidak ada ")
                    
                #end if 
                
            else:
                print(f"Direktori '{Dir2}' tidak ditemukan.")
            #end if 
        cap.release()
        cv2.destroyAllWindows()
        
    def LoadDataSet(self, ListKelas):
        X = []
        y = [] 
        LB = np.eye(len(ListKelas))
        for index,kl in enumerate(ListKelas) : 
            
            ListFit= self.EkstraksiFiturDataSet(kl,FILEEXT=".jpg",SaveFile = False)
            for  fit in ListFit: 
                
                X.append(fit)
                y.append(LB[index,:])
                
                
        X =np.array(X)
        y=np.array(y)
        print(X.shape )
        
        return X,y 
        
    def EkstraksiFiturDataSet(self, NamaKelas,FILEEXT=".jpg",SaveFile = False):        
        DIR= os.path.join(self.DirektoriDataSet,NamaKelas )
        subdirs = self.list_subdirectories(DIR)
        
        ListFitur=[] 
        # Loop melalui setiap sub-direktori
        for subdir in subdirs:
            subdir_path = os.path.join(DIR, subdir)
            KelasFit=[] 
            # Loop untuk membaca semua file di dalam sub-direktori
            RawFitBef  =[]
            for filename in os.listdir(subdir_path):
                print(filename)
                # Periksa apakah file memiliki ekstensi JPG
                if filename.lower().endswith(FILEEXT):
                    file_path = os.path.join(subdir_path, filename)
                    image = cv2.imread(file_path)
                    if SaveFile : 
                        self.Proses.EkstraksiFitur(image,file_path)
                    else: 
                        RawFit = self.Proses.EkstraksiFitur(image)
                        Fitur = self.Proses.ProsesFitur(RawFit,RawFitBef)
                        
                        RawFitBef =RawFit
                        if len(Fitur) :
                            KelasFit.append(Fitur)
                        #end of 
                    #end if 
                #end if
            KelasFit = np.array(KelasFit)
            w = self.wr(KelasFit, self.LebarWindow, self.WIndowStep)
                
       

            ListFitur.append(w)
        
        return ListFitur 
    def list_subdirectories(self,DIR):
        # Mengecek apakah DIR adalah direktori yang valid
        if not os.path.isdir(DIR):
            print(f"'{DIR}' bukan direktori yang valid.")
            return []
    
        # Daftar untuk menyimpan nama sub-direktori
        subdirectories = []
    
        # Loop untuk memeriksa setiap item di dalam DIR
        for item in os.listdir(DIR):
            # Membuat path lengkap dari item
            item_path = os.path.join(DIR, item)
    
            # Jika item adalah direktori, tambahkan ke daftar subdirektori
            if os.path.isdir(item_path):
                subdirectories.append(item)
    
        return subdirectories
    def ListKelas(self):
        try:
            folders = [item for item in os.listdir(self.DirektoriDataSet ) if os.path.isdir(os.path.join(self.DirektoriDataSet , item))]
            return folders
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error: {e}")
            return []
    def TrainingGesture(self,Training ,Kelas):
        X,y = self.LoadDataSet(Kelas)
        self.model =Training(X,y)
        
        
        
        
        self.model.save(self.NamaModel)
        
        

    
    def KlasifikasiGesture(self ,labels):  
        Proses = self.Proses
        self.model = tf.keras.models.load_model(self.NamaModel)
        
        SR =self.SaveRate 
        cap = cv2.VideoCapture(self.NoKamera)
        # Mengecek apakah webcam berhasil dibuka
        if not cap.isOpened():
            print("Tidak dapat mengakses webcam")
            sys.exit()
        
        # Menghitung waktu awal
        BefTime= datetime.now()
        L = [] 
        RawFitBef= [] 
        
        # Loop utama
        c =0 
        while True:
            # Membaca frame dari webcam
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame")
                break
        
            frame = cv2.flip(frame, 1) 
            FrameSimpan = frame.copy() 
            frame = Proses.Capture(frame)            
            # Mendapatkan waktu saat ini
            current_time = datetime.now()
        
            
            if (current_time - BefTime).total_seconds() > 1 / SR: 
                
                RawFit = self.Proses.EkstraksiFitur(FrameSimpan)
                if len(L)==0:
                   fit = self.Proses.ProsesFitur(RawFit, RawFit)
                else:
                   fit = self.Proses.ProsesFitur(RawFit, RawFitBef )
                   
                L.append(fit)
                RawFitBef = RawFit.copy() 
                if len(L)>self.JumlahFrame: 
                    c=c+1
                    L.pop(0)
                    X = np.array(L)
                    w = self.wr(X, self.LebarWindow, self.WIndowStep)
                    Y =[] 
                    Y.append(w)
                    Y = np.array(Y)
                    p = self.model.predict(Y,verbose=0)
                 

                    # Get the predictions
                    p = self.model.predict(Y, verbose=0)
                    
                    # Assuming `p` is a probability distribution, get the index of the max probability
                    predicted_classes = np.argmax(p, axis=1)
                    
                    # Map the predictions to labels
                    predicted_labels = [labels[i] for i in predicted_classes]
                    
                    # Print the labeled predictions
                    print("Predicted labels:", predicted_labels)
                                            
                                            
                
                BefTime = current_time
                
            
                # Menampilkan frame di jendela 'Webcam'
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #end if 
        #end while 
        cap.release()
        cv2.destroyAllWindows()

        
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, TimeDistributed, LeakyReLU

from tensorflow.keras.layers import Dropout
   


def Training(X,y) :
    nKelas = y.shape[1] 
    model = Sequential()
    sp=(X.shape[1], X.shape[2], X.shape[3])
    # First Conv1D layer with LeakyReLU activation
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3), 
                              input_shape=sp))
    model.add(TimeDistributed(LeakyReLU(alpha=0.5)))

    # Second Conv1D layer with LeakyReLU activation
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3)))
    model.add(TimeDistributed(LeakyReLU(alpha=0.5)))

    # Flatten layer
    model.add(TimeDistributed(Flatten()))

    # LSTM layer with tanh activation (default)
    model.add(LSTM(units=64))

    # Output layer with softmax for multi-class classification
    model.add(Dense(1000, activation='linear'))
    
    model.add(Dropout(0.5)) 
    model.add(Dense(1000, activation='linear'))
    
              
    
    model.add(Dense(nKelas, activation='softmax'))
    

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Pelatihan model
    model.fit(X, y, epochs=1000, batch_size=1, validation_split=0.2,shuffle=True)
    # Evaluasi model
    loss, accuracy = model.evaluate(X, y)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return model
   
labels = ["Kanan", "Kiri","Lurus","Maju","Mundur"]        
Gesture = CLGesture(CLFiturFaceGesture())


import pygame

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

def run_menu():
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
                elif event.key == pygame.K_9:
                    pygame.quit()
                    sys.exit()

# Display instructions
print("Press the following keys for each option:")
for i, option in enumerate(menu_options):
    print(f"{i + 1}: {option}")

# Run the menu
run_menu()