import mediapipe as mp
import cv2
import numpy as np 
import math 
class CLFiturGesture:
    def __init__(self):
      self.mp_hands = mp.solutions.hands
      self.mp_drawing = mp.solutions.drawing_utils

      # Konfigurasi untuk mendeteksi tangan kanan saja
      self.hands = self.mp_hands.Hands(
          static_image_mode=False,
          max_num_hands=1,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5
      )

    def ProsesFitur(self,RawFit,RawFitBef,NamaFile=""): 
        NamaFile = NamaFile.replace(".jpg", "_fit.png")
        if len(NamaFile)>0: 
            cv2.imwrite(NamaFile , RawFit)
        
        fit =RawFit
        
        
        
        return fit
    
    def PostProsesFitur(self,RawFit): 
        fit = RawFit
        return fit
    
    
    
    def EkstraksiFitur(self, frame, NamaFile=""):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Inisialisasi array kosong 128x128x3 dengan tipe uint8
        fit = np.zeros((128, 128, 3), dtype=np.uint8)
    
        # Proses frame untuk mendeteksi tangan
        results = self.hands.process(frame_rgb)
    
        # Cek apakah ada hasil deteksi tangan
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Cek apakah tangan yang terdeteksi adalah tangan kanan
                if handedness.classification[0].label == "Right":
                    # Menggambar landmark dan koneksi pada tangan
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Mengambil koordinat minimum dan maksimum dari landmark untuk cropping
                    h, w, c = frame.shape
                    x_min = w
                    x_max = 0
                    y_min = h
                    y_max = 0
    
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        x_max = max(x_max, x)
                        y_min = min(y_min, y)
                        y_max = max(y_max, y)
    
                    # Menambahkan padding untuk cropping
                    padding = 20
                    x_min = max(x_min - padding, 0)
                    y_min = max(y_min - padding, 0)
                    x_max = min(x_max + padding, w)
                    y_max = min(y_max + padding, h)
    
                    # Membuat tinggi dan lebar cropping sama (persegi)
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    box_size = max(box_width, box_height)
    
                    # Pastikan cropping tetap berada dalam batas gambar
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2
    
                    x_min_square = max(x_center - box_size // 2, 0)
                    y_min_square = max(y_center - box_size // 2, 0)
                    x_max_square = min(x_min_square + box_size, w)
                    y_max_square = min(y_min_square + box_size, h)
    
                    # Crop area tangan berbentuk persegi
                    hand_crop_square = frame[y_min_square:y_max_square, x_min_square:x_max_square]
    
                    # Ubah ukuran hasil crop menjadi 128 x 128
                    hand_crop_resized = cv2.resize(hand_crop_square, (128, 128)).astype(np.float32)
                    
                    fit =  hand_crop_resized.copy()
                    break  # Keluar dari loop setelah mendeteksi tangan kanan
        return fit

    
    def Close(self):
        #self.face_mesh.close()
        return 
    

        
    def Capture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses frame untuk mendeteksi tangan
        results = self.hands.process(frame_rgb)

        # Cek apakah ada hasil deteksi tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Menggambar landmark dan koneksi pada tangan
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame.copy()

