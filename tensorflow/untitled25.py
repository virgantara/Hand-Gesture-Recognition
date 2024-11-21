import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Hands dan Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi deteksi tangan
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break
    frame = cv2.flip(frame, 1)  # Membalik gambar secara horizontal




    # Konversi warna frame dari BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Proses deteksi tangan
    result = hands.process(frame_rgb)

    # Variabel untuk menyimpan gambar yang dipotong
    cropped_hand = None

    # Jika ada tangan yang terdeteksi
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Periksa apakah tangan yang terdeteksi adalah tangan kanan
            if handedness.classification[0].label == 'Right':
                # Dapatkan koordinat bounding box (pojok kiri atas dan kanan bawah)
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                # Hitung nilai minimum dan maksimum untuk mendapatkan bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Konversi koordinat relatif ke koordinat piksel
                h, w, _ = frame.shape
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                
                # Hitung tinggi dan lebar bounding box
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Menentukan sisi terpanjang untuk membuat bounding box menjadi persegi
                box_side = max(box_width, box_height)

                # Tentukan koordinat bounding box persegi
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_min_square = max(0, x_center - box_side // 2)
                x_max_square = min(w, x_center + box_side // 2)
                y_min_square = max(0, y_center - box_side // 2)
                y_max_square = min(h, y_center + box_side // 2)

                # Gambar landmark tangan dan koneksi pada frame
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Memotong gambar berdasarkan bounding box persegi setelah diberi gambar landmark
                cropped_hand = frame[y_min_square:y_max_square, x_min_square:x_max_square]

                # Gambar bounding box persegi di sekitar tangan pada frame asli
                cv2.rectangle(frame, (x_min_square, y_min_square), (x_max_square, y_max_square), (0, 255, 0), 2)

                # Jika sudah mendeteksi satu tangan kanan, hentikan loop
                break

    # Jika tidak ada tangan yang terdeteksi atau hasil crop tidak valid, buat gambar hitam ukuran 30x30
    if cropped_hand is None or cropped_hand.size == 0:
        cropped_hand = np.zeros((30, 30, 3), dtype=np.uint8)

    # Tampilkan gambar yang telah dipotong
    cv2.imshow('Cropped Hand with Landmarks', cropped_hand)

    # Tampilkan frame asli dengan bounding box
    cv2.imshow('Hand Detection with Square Bounding Box', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
hands.close()
