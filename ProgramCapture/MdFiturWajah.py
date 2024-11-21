import mediapipe as mp
import cv2
import numpy as np 
import math 
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

    def ProsesFitur(self,RawFit,RawFitBef,NamaFile=""): 
        fit =( RawFit.copy() -RawFitBef.copy() )*10
        fit=np.concatenate( (fit, RawFit))*10
        NamaFile =NamaFile.replace(".jpg", "_new_fit.txt")
        if NamaFile :  
            np.savetxt(NamaFile, fit,delimiter=",")
        return fit
    
    def PostProsesFitur(self,RawFit): 
        fit = RawFit
        return fit
    
    def get_normalized_landmarks(self, face_landmarks, img_width, img_height):
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263
        NOSE_TIP = 1
        LEFT_MOUTH_CORNER = 61
        RIGHT_MOUTH_CORNER = 291
    
        left_eye = np.array([
            face_landmarks.landmark[LEFT_EYE_OUTER].x * img_width,
            face_landmarks.landmark[LEFT_EYE_OUTER].y * img_height
        ])
    
        right_eye = np.array([
            face_landmarks.landmark[RIGHT_EYE_OUTER].x * img_width,
            face_landmarks.landmark[RIGHT_EYE_OUTER].y * img_height
        ])
    
        nose_tip = np.array([
            face_landmarks.landmark[NOSE_TIP].x * img_width,
            face_landmarks.landmark[NOSE_TIP].y * img_height
        ])
    
        left_mouth = np.array([
            face_landmarks.landmark[LEFT_MOUTH_CORNER].x * img_width,
            face_landmarks.landmark[LEFT_MOUTH_CORNER].y * img_height
        ])
    
        right_mouth = np.array([
            face_landmarks.landmark[RIGHT_MOUTH_CORNER].x * img_width,
            face_landmarks.landmark[RIGHT_MOUTH_CORNER].y * img_height
        ])
    
        # Calculate midpoint between the eyes
        eye_midpoint = (left_eye + right_eye) / 2
    
        # Calculate the reference distance (Euclidean distance between eye midpoint and nose tip)
        reference_distance = np.linalg.norm(eye_midpoint - nose_tip)
        # Ensure reference_distance is at least 1 to avoid too-small values
        if reference_distance < 1:
            reference_distance = 1
        # Calculate mean_x and mean_y
        mean_x = np.mean([left_eye[0], right_eye[0], nose_tip[0], left_mouth[0], right_mouth[0]])
        mean_y = np.mean([left_eye[1], right_eye[1], nose_tip[1], left_mouth[1], right_mouth[1]])
    
        # Create the normalized array
        normalized_array = np.array([
            left_eye[0] - mean_x, left_eye[1] - mean_y,
            right_eye[0] - mean_x, right_eye[1] - mean_y,
            nose_tip[0] - mean_x, nose_tip[1] - mean_y,
            left_mouth[0] - mean_x, left_mouth[1] - mean_y,
            right_mouth[0] - mean_x, right_mouth[1] - mean_y
        ]) / reference_distance

        return normalized_array


    



    def EkstraksiFitur(self, frame, NamaFile=""):
       frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
       result = self.face_mesh.process(frame_rgb)
       h, w, _ = frame.shape    
       fit=np.zeros((10,), dtype=np.float32)
       
       if result.multi_face_landmarks:
           
          
            face_landmarks = result.multi_face_landmarks[0]
            if face_landmarks is None:
                  # Return a blank 30x30x3 image if no face landmarks are provided
                return fit
            fit = self.get_normalized_landmarks(face_landmarks,h,w)
       return fit
       


    
    
    
    def Close(self):
        self.face_mesh.close()
    
    def draw_face_points(self, frame):
        # Konversi frame ke RGB untuk diproses oleh MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Titik-titik penting di wajah
                left_eye = face_landmarks.landmark[33]    # Ujung mata kiri
                right_eye = face_landmarks.landmark[263]  # Ujung mata kanan
                nose_tip = face_landmarks.landmark[1]     # Ujung hidung
                left_mouth = face_landmarks.landmark[61]  # Ujung mulut kiri
                right_mouth = face_landmarks.landmark[291] # Ujung mulut kanan
                mid_eye = ((left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2)

                # Fungsi untuk konversi koordinat normalisasi ke piksel
                def to_pixel(landmark, frame_shape):
                    h, w, _ = frame_shape
                    return int(landmark.x * w), int(landmark.y * h)

                # Konversi landmark ke koordinat piksel
                left_eye_px = to_pixel(left_eye, frame.shape)
                right_eye_px = to_pixel(right_eye, frame.shape)
                nose_tip_px = to_pixel(nose_tip, frame.shape)
                left_mouth_px = to_pixel(left_mouth, frame.shape)
                right_mouth_px = to_pixel(right_mouth, frame.shape)
                mid_eye_px = int(mid_eye[0] * frame.shape[1]), int(mid_eye[1] * frame.shape[0])

                # Gambar lingkaran pada setiap titik
                cv2.circle(frame, left_eye_px, 5, (0, 255, 0), -1)     # Lingkaran pada ujung mata kiri
                cv2.circle(frame, right_eye_px, 5, (0, 255, 0), -1)    # Lingkaran pada ujung mata kanan
                cv2.circle(frame, nose_tip_px, 5, (255, 0, 0), -1)     # Lingkaran pada ujung hidung
                cv2.circle(frame, left_mouth_px, 5, (0, 0, 255), -1)   # Lingkaran pada ujung mulut kiri
                cv2.circle(frame, right_mouth_px, 5, (0, 0, 255), -1)  # Lingkaran pada ujung mulut kanan
                cv2.circle(frame, mid_eye_px, 5, (255, 255, 0), -1)    # Lingkaran pada titik tengah antara kedua mata

        return frame
        
    def Capture(self, frame):
        frame = self.draw_face_points(frame)        
        return frame.copy()

