
import cv2
import sys
import shutil
import numpy as np
import os
import seaborn as sns
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
import sounddevice as sd
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class CLGesture:
    def __init__(self, Proses,DataSet="DataSet"):
        self.Proses = Proses
        self.Kelas = []
        self.SaveRate = 15
        self.WaktuJeda = 4
        self.WaktuRekamGesture = 4
        self.DirektoriDataSet = DataSet
        self.JumlahFrame = 20
        self.NamaModel ="model.h5"
        self.NoKamera = 0 
        self.LebarWindow=20 
        self.WIndowStep = 4
        self.Epoh  =100 
        
        
    
        
        
        
        
        
        
    def SetKamera(self,NoKamera):
        self.NoKamera =NoKamera
        
    def TesWebCam(self):
        # Inisialisasi webcam dan kelas FaceLandmarkPoints
        cap = cv2.VideoCapture(self.NoKamera)
        
    
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Gagal membaca dari webcam.")
                break
    
            # Tampilkan frame
            cv2.imshow("Face Landmark Points", frame)
    
            # Keluar dengan menekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Bersihkan resource
        cap.release()
        cv2.destroyAllWindows()



    # Fungsi untuk menghasilkan dan memainkan suara beep
    def beep_twice(self):
        frequency = 1000  # Frekuensi suara dalam Hz
        duration = 0.2    # Durasi suara dalam detik
    
        # Buat array untuk suara beep
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = 1* np.sin(2 * np.pi * frequency * t)
    
        # Mainkan beep pertama
        sd.play(wave, samplerate=sample_rate)
        sd.wait()  # Tunggu sampai beep selesai
    
        # Jeda 200 ms
        time.sleep(0.2)
    
        # Mainkan beep kedua
        sd.play(wave, samplerate=sample_rate)
        sd.wait()  # Tunggu sampai beep selesai
    
    # Panggil fungsi beep dua kali
    
    def beep(self):
        frequency = 800  # Frekuensi suara dalam Hz
        duration = 0.2    # Durasi suara dalam detik
    
        # Buat array untuk suara beep
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = 1* np.sin(2 * np.pi * frequency * t)
    
        # Mainkan beep pertama
        sd.play(wave, samplerate=sample_rate)
        sd.wait()  # Tunggu sampai beep selesai
    
 
    # Panggil fungsi beep dua kali
    
            
    
    def Hitung(self,dir_path = 'dir1'):
        # Set the path of the main directory
        
        
        # Check if the path exists and is a directory
        if not os.path.isdir(dir_path):
            
            return 0
        
        # Count the number of directories
        dir_count = sum(os.path.isdir(os.path.join(dir_path, item)) for item in os.listdir(dir_path))
        return dir_count

        
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


    def HitungDirektoriSetiapKelas(self, Kelas):
        Dir1 =self.DirektoriDataSet
        print("===========================================")
        
        for kelas in Kelas:
           # Create the full path for each class directory inside Dir1
           kelas_path = os.path.join(Dir1, kelas)
           
           # Check if the directory exists
           if os.path.isdir(kelas_path):
               # Count the number of subdirectories in the current class directory
               dir_count = sum(os.path.isdir(os.path.join(kelas_path, item)) for item in os.listdir(kelas_path))
               print(f"Jumlah direktori di dalam '{kelas_path}': {dir_count}")
           else:
               print(f"Jumlah direktori di dalam '{kelas_path}': 0")
        print("------------------------------------------")
        
 
    
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
        DIR1Simpan= DIR1
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
        countsimpan = 0 
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
           
            
            nn =35
            # Menampilkan counter sebelum penyimpanan dimulai
            if not saving_started:
                ndir = self.Hitung(DIR1)
                
                # Menghitung waktu berlalu sebelum penyimpanan dimulai
                elapsed_time = (current_time - start_time).total_seconds()
                # Menampilkan penghitung waktu pada frame sebelum mulai menyimpan
                cv2.putText(frame, f"{NamaKelas}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


                cv2.putText(frame, f"Starting in: {int(N - elapsed_time)}", (50, 50+nn*1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Jumlah Data Set : {ndir}", (50, 50+nn*2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(frame, f"Jumlah Data Disimpan: {countsimpan}", (50, 50+nn*3),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                # Jika waktu jeda telah tercapai, mulai penyimpanan
                if elapsed_time >= N:
                    self.beep_twice()
                    saving_started = True
                    # Reset waktu untuk mulai penghitungan elapsed time dari awal penyimpanan
                    start_capture_time = current_time
                    BefTime = current_time  # Set waktu awal penyimpanan
                   
                    
                #end if             
            else: 
            
                # Menghitung waktu berlalu sejak penyimpanan dimulai
                elapsed_capture_time = (current_time - start_capture_time).total_seconds()
                cv2.putText(frame, f"{NamaKelas}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
                # Menampilkan jumlah file yang disimpan pada frame setelah penyimpanan dimulai
                cv2.putText(frame, f"Files saved: {file_count}", (50, 50+nn*1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Elapsed Time: {int(elapsed_capture_time)}s", (50, 50+nn*2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Count: {countsimpan}", (50, 50+nn*3),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                # Mengecek apakah waktunya untuk menyimpan frame
                if (current_time - BefTime).total_seconds() > 1 / SR:
                    timestamp = current_time.strftime('%y%m%d%H%M%S%f')[:-3]  # Format YYMMDDHHmmSSMS
                 
                    if file_count==0:
                        if countsimpan==0: 
                            Dir2 = os.path.join(DIR1, f"{timestamp}_START")
                        else:
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
                    self.beep()
                    print("Waktu penyimpanan telah tercapai, reset...")
                    saving_started = False
                    start_time = current_time
                    countsimpan  =countsimpan +1 
                    
                    file_count = 0
                # Menampilkan frame di jendela 'Webcam'
            cv2.imshow(NamaKelas, frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or countsimpan>10:
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
        
    def LoadDataSet(self ):
        ListKelas = self.Kelas
        X = []
        y = [] 
        LB = np.eye(len(ListKelas))
        for index,kl in enumerate(ListKelas) : 
            
            ListFit= self.EkstraksiFiturDataSet(kl,FILEEXT=".jpg")
            for  fit in ListFit: 
                
                X.append(fit)
                y.append(LB[index,:])
                
                
        X =np.array(X)
        y=np.array(y)
        print(X.shape )
        
        return X,y 
        
    def EkstraksiFiturDataSet(self, NamaKelas,FILEEXT=".jpg"):        
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
           
                # Periksa apakah file memiliki ekstensi JPG
                if filename.lower().endswith(FILEEXT):
                    print(filename)
                    file_path = os.path.join(subdir_path, filename)
                    image = cv2.imread(file_path)
                    
                    RawFit = self.Proses.EkstraksiFitur(image,file_path)
                    if len(RawFitBef)>0: 
                        Fitur = self.Proses.ProsesFitur(RawFit,RawFitBef,file_path)
                        KelasFit.append(Fitur)
                    #end if
                    RawFitBef =RawFit
                #end if
            #end for 
            
            KelasFit = self.Proses.PostProsesFitur(KelasFit)
            KelasFit = np.array(KelasFit)
            ListFitur.append(KelasFit )
        
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
    
    
    def Training(self,Model,X, y):
        Epoh = self.Epoh
        Kelas = self.Kelas 
    
        model  = Model(X,y)
            
            
        
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # Train the model
        history = model.fit(X_train, y_train, epochs=Epoh, batch_size=32, validation_data=(X_test, y_test),
                            shuffle=True, callbacks=[early_stopping])
        # Plot training history and confusion matrix
        
        # Generate directory name with current timestamp (YYMMDDHHmmssms format)
        dir1 = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3]
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
        cm_df = pd.DataFrame(cm, index=Kelas, columns=Kelas)
        cm_csv_path = os.path.join(save_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_csv_path)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Kelas, yticklabels=Kelas)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.show()  # Display the plot
        print(f"Plots and CSV files saved in directory: {save_dir}")
        model.save(self.NamaModel)
        


    
    
    def TrainingGesture(self,Model ):
        X,y = self.LoadDataSet()
        self.Training(Model,X,y)
        
        
        
        
        
    
    def KlasifikasiGesture(self ):
        labels = self.Kelas
        predicted_labels=""
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
                    Y =[] 
                    Y.append( X )
                    Y = np.array(Y)
                    
                    p = self.model.predict(Y,verbose=0)
                    # Assuming `p` is a probability distribution, get the index of the max probability
                    predicted_classes = np.argmax(p, axis=1)
                    predicted_labels = [labels[i] for i in predicted_classes]
        
            
                    
                   
                                            
                BefTime = current_time
                
            cv2.putText(frame, f'Prediction: {predicted_labels}', 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)
        
                # Menampilkan frame di jendela 'Webcam'
            cv2.imshow('Webcam', frame)
            print(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #end if 
        #end while 
        cap.release()
        cv2.destroyAllWindows()
