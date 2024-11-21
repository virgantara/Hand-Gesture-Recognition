import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os

# Inisialisasi Mediapipe Hands dan Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi tangan
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Buka webcam
cap = cv2.VideoCapture(0)

# Default ukuran window
cv2.namedWindow('Hand Landmark Detection', cv2.WINDOW_NORMAL)

is_maximized = False  # Status maximize/minimize

# Variabel untuk kontrol waktu dan counter
last_save_time = datetime.now()
counter = 0

# Path direktori untuk menyimpan frame
save_dir = "dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Buat direktori jika belum ada
    print(f"Direktori '{save_dir}' berhasil dibuat.")

def get_time_str():
    """Mengembalikan string waktu dalam format hhmmssms."""
    return datetime.now().strftime("%H%M%S%f")[:-3]  # Mengambil hingga milidetik

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam")
        break

    # Konversi ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Salin frame asli sebelum digambar landmark
    original_frame = frame.copy()

    # Buat dummy image dengan background hitam
    dummy_frame = np.zeros_like(frame)

    # Proses frame
    results = hands.process(frame_rgb)

    # Jika ada tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan di frame asli
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Gambar landmark tangan di dummy frame
            mp_drawing.draw_landmarks(dummy_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hitung selisih waktu
    current_time = datetime.now()
    time_difference = (current_time - last_save_time).total_seconds()

    # Simpan otomatis jika lebih dari 1 detik
    if time_difference > 1:
        counter += 1  # Increment counter
        time_str = get_time_str()

        # Tambahkan counter ke frame
        cv2.putText(original_frame, f"{counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(dummy_frame, f"{counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Simpan frame di direktori "dataset"
        cv2.imwrite(os.path.join(save_dir, f"{time_str}.jpg"), original_frame)  # Simpan frame asli
        cv2.imwrite(os.path.join(save_dir, f"{time_str}.png"), frame)          # Simpan frame dengan landmark
        cv2.imwrite(os.path.join(save_dir, f"{time_str}.bmp"), dummy_frame)    # Simpan frame dummy

        print(f"Frame sebelum landmark disimpan: {save_dir}/{time_str}.jpg")
        print(f"Frame dengan landmark disimpan: {save_dir}/{time_str}.png")
        print(f"Frame dummy dengan landmark disimpan: {save_dir}/{time_str}.bmp")

        # Update waktu terakhir menyimpan
        last_save_time = current_time

    # Tampilkan frame
    cv2.imshow('Hand Landmark Detection', frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Berhenti jika tombol 'q' ditekan
        break
    elif key == ord('m'):  # Maksimalkan/minimalkan window dengan tombol 'm'
        if is_maximized:
            cv2.setWindowProperty('Hand Landmark Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            is_maximized = False
        else:
            cv2.setWindowProperty('Hand Landmark Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            is_maximized = True

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
hands.close()
