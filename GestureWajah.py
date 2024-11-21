# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:53:44 2024

@author: visikom2023
"""

import cv2
import sys
import shutil
import numpy as np
import os
from datetime import datetime
import mediapipe as mp

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
        
    def EkstraksiFitur(self, frame, NamaFile=None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = self.face_mesh.process(frame_rgb)
        cropped_face = None

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Ambil koordinat bounding box berdasarkan landmark wajah
                x_coords = [landmark.x for landmark in face_landmarks.landmark]
                y_coords = [landmark.y for landmark in face_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                h, w, _ = frame.shape
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                
                # Membuat bounding box persegi untuk crop wajah
                box_width = x_max - x_min
                box_height = y_max - y_min
                box_side = max(box_width, box_height)
                
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_min_square = max(0, x_center - box_side // 2)
                x_max_square = min(w, x_center + box_side // 2)
                y_min_square = max(0, y_center - box_side // 2)
                y_max_square = min(h, y_center + box_side // 2)

                # Menggambar landmark dan koneksi pada wajah
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                cropped_face = frame[y_min_square:y_max_square, x_min_square:x_max_square]
                cv2.rectangle(frame, (x_min_square, y_min_square), (x_max_square, y_max_square), (0, 255, 0), 2)
                break

        # Jika tidak ada wajah terdeteksi, buat gambar hitam ukuran 30x30
        if cropped_face is None or cropped_face.size == 0:
            cropped_face = np.zeros((30, 30, 3), dtype=np.uint8)
        cropped_face = cv2.resize(cropped_face, (256, 256))
        Fitur = cropped_face.astype("float32") / 255
        
        if NamaFile:
            NamaFile = NamaFile.replace(".jpg", ".png")
            cv2.imwrite(NamaFile, Fitur * 255)
        else:
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
    
    def __del__(self):
        print("Exit")
        cv2.destroyAllWindows()

    
    def CaptureGesture(self, NamaKelas, SaveRate=None, JumlahFrame=None, WaktuJeda=None):
        Proses = self.Proses
        SR = SaveRate if SaveRate else self.SaveRate
        N = WaktuJeda if WaktuJeda else self.WaktuJeda
        JumlahFrame = JumlahFrame if JumlahFrame else self.JumlahFrame
        
        if not os.path.exists(self.DirektoriDataSet):
            os.makedirs(self.DirektoriDataSet)
        DIR1 = os.path.join(self.DirektoriDataSet, NamaKelas)
        
        if not os.path.exists(DIR1):
            os.makedirs(DIR1)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Tidak dapat mengakses webcam")
            sys.exit()
        
        start_time = datetime.now()
        file_count = 0
        saving_started = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame")
                break
            
            frame = cv2.flip(frame, 1)
            FrameSimpan = frame.copy()
            
            if Proses:
                frame = Proses.Capture(frame)
            
            current_time = datetime.now()
            
            if not saving_started:
                elapsed_time = (current_time - start_time).total_seconds()
                cv2.putText(frame, f"Starting in: {int(N - elapsed_time)}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if elapsed_time >= N:
                    saving_started = True
                    start_capture_time = current_time
                    BefTime = current_time
            
            else:
                elapsed_capture_time = (current_time - start_capture_time).total_seconds()
                cv2.putText(frame, f"Files saved: {file_count}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Elapsed Time: {int(elapsed_capture_time)}s", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                
                if (current_time - BefTime).total_seconds() > 1 / SR:
                    timestamp = current_time.strftime('%y%m%d%H%M%S%f')[:-3]
                    filename = os.path.join(DIR1, f"{timestamp}.jpg")
                    cv2.imwrite(filename, FrameSimpan)
                    print(f"Frame disimpan: {filename}")
                    file_count += 1
                    BefTime = current_time
                
                if file_count >= JumlahFrame:
                    print("Jumlah frame penyimpanan tercapai, reset...")
                    saving_started = False
                    start_time = current_time
                    file_count = 0
            
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()




Klasifikasi = CLGesture(CLFiturFaceGesture())
Klasifikasi.CaptureGesture("FaceSample")
