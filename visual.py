import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

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

    # Tampilkan frame
    cv2.imshow('Hand Landmark Detection', frame)

    # Simpan frame asli, frame dengan landmark, dan dummy frame
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Tekan 's' untuk menyimpan frame
        time_str = get_time_str()
        cv2.imwrite(f"{time_str}.jpg", original_frame)  # Simpan frame asli
        cv2.imwrite(f"{time_str}.png", frame)          # Simpan frame dengan landmark
        cv2.imwrite(f"{time_str}.bmp", dummy_frame)    # Simpan frame dummy
        print(f"Frame sebelum landmark disimpan: {time_str}.jpg")
        print(f"Frame dengan landmark disimpan: {time_str}.png")
        print(f"Frame dummy dengan landmark disimpan: {time_str}.bmp")

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
