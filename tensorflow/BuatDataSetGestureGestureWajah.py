import cv2
import sys
import shutil
import numpy as np 
import shutil
import os
from datetime import datetime

import mediapipe as mp



class CLFiturGestureWajah:
    # Constructor (initializer)
    def __init__(self):
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils


    def EkstraksiFitur(self, frame, NamaFile=None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face landmarks
        result = self.face_mesh.process(frame_rgb)
        cropped_face = None
    
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Get bounding box coordinates from face landmarks
                x_coords = [landmark.x for landmark in face_landmarks.landmark]
                y_coords = [landmark.y for landmark in face_landmarks.landmark]
    
                # Calculate min and max values for bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert relative coordinates to pixel coordinates
                h, w, _ = frame.shape
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                
                # Calculate width and height of bounding box
                box_width = x_max - x_min
                box_height = y_max - y_min
    
                # Determine the longest side to make a square bounding box
                box_side = max(box_width, box_height)
    
                # Define square bounding box coordinates
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_min_square = max(0, x_center - box_side // 2)
                x_max_square = min(w, x_center + box_side // 2)
                y_min_square = max(0, y_center - box_side // 2)
                y_max_square = min(h, y_center + box_side // 2)
    
                # Draw landmarks and connections on the frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    self.mp_face_mesh.FACE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )
    
                # Crop the square face region
                cropped_face = frame[y_min_square:y_max_square, x_min_square:x_max_square]
                cv2.rectangle(frame, (x_min_square, y_min_square), (x_max_square, y_max_square), (0, 255, 0), 2)
                break
    
        # If no face detected or crop is invalid, create a black 30x30 image
        if cropped_face is None or cropped_face.size == 0:
            cropped_face = np.zeros((30, 30, 3), dtype=np.uint8)
        cropped_face = cv2.resize(cropped_face, (256, 256))
        
        cropped_face = cropped_face.astype("float32")
        Fitur = cropped_face / 255
        return Fitur    
             
            
            
            
            
            
    def Close(self):
        self.hands.close() 
        
    import cv2
import mediapipe as mp

class FaceLandmarkExtractor:


    def Capture(self, frame):
        frameProses = frame.copy() 
        rgb_frame = cv2.cvtColor(frameProses, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        # Check if FACE_CONNECTIONS exists
        if hasattr(self.mp_face_mesh, 'FACE_CONNECTIONS'):
            connections = self.mp_face_mesh.FACE_CONNECTIONS
        else:
            connections = None  # Set to None if FACE_CONNECTIONS is unavailable

        # Draw face landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if connections:
                    # Draw landmarks with connections if available
                    self.mp_drawing.draw_landmarks(
                        frameProses,
                        face_landmarks,
                        connections,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )
                else:
                    # Draw only landmark points if FACE_CONNECTIONS is not available
                    self.mp_drawing.draw_landmarks(
                        frameProses,
                        face_landmarks,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        
        return frameProses.copy()

        


class CLGesture() :
    def __init__(self, Proses):
        self.Proses = Proses
        self.Kelas = []
        self.SaveRate= 15
        self.WaktuJeda = 4
        self.WaktuRekamGesture=4
        self.DirektoriDataSet = "DataSet"
        self.JumlahFrame = 40 
    
        
    def __del__(self):
        print("Exit")
        
        self.Proses.Close() 
          
        
    def SetKelas(self ,Kelas):
        self.Kelas = Kelas 
        
        

    def ListKelas(self):
        
        try:
            folders = [item for item in os.listdir(self.DirektoriDataSet ) if os.path.isdir(os.path.join(self.DirektoriDataSet , item))]
            return folders
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error: {e}")
            return []

    
    def CaptureGesture(self ,NamaKelas,SaveRate =None,JumlahFrame = None ,WaktuJeda= None):  
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
        cap = cv2.VideoCapture(0)
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
                if file_count >= JumlahFrame:
                        
                    print("Waktu penyimpanan telah tercapai, reset...")
                    saving_started = False
                    start_time = current_time
                    file_count = 0
            
                # Menampilkan frame di jendela 'Webcam'
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #end if 
        #end while 
            # Cek apakah direktori ada
        if os.path.exists(Dir2):
            # Hapus direktori beserta isinya
            if 0< file_count <JumlahFrame :  
                shutil.rmtree(Dir2)
            #end if 
            print(f"Direktori '{Dir2}' dan seluruh isinya telah dihapus.")
        else:
            print(f"Direktori '{Dir2}' tidak ditemukan.")
        #end if 
        cap.release()
        cv2.destroyAllWindows()
        
        
        
    
    
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
    
    def EkstraksiFiturDataSet(self, NamaKelas,FILEEXT=".jpg",SaveFile = False):
        

        DIR= os.path.join(self.DirektoriDataSet,NamaKelas )


        
        subdirs = self.list_subdirectories(DIR)
        ListFitur=[] 
        
    
        # Loop melalui setiap sub-direktori
        for subdir in subdirs:
            subdir_path = os.path.join(DIR, subdir)
            subFit=[] 
    
            # Loop untuk membaca semua file di dalam sub-direktori
            for filename in os.listdir(subdir_path):
                # Periksa apakah file memiliki ekstensi JPG
                if filename.lower().endswith(FILEEXT):
                    file_path = os.path.join(subdir_path, filename)
                    print(file_path )
                    image = cv2.imread(file_path)
                    if SaveFile : 
                        self.Proses.EkstraksiFitur(image,file_path)
                    else: 
                        fit = self.Proses.EkstraksiFitur(image)
                        subFit.append(fit)
            ListFitur.append(subFit)
        
        return ListFitur 
                        
                        
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
        print(X.shape)
        y=np.array(y)
        return X,y 
    def Training(self):
        X,y = self.Proses.LoadDataSet(self.ListKelas())
    def HapusDataset(self,NamaKelas):
        dir_path = os.join(self.DirektoriDataSet,NamaKelas ) 
        shutil.rmtree(dir_path)
        
        
        

        
    
                
            
            
    
    
Gesture = CLGesture(CLFiturGestureWajah())

Gesture.CaptureGesture("COBA7")
