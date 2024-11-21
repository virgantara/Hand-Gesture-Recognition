import cv2
import sys
import os
from datetime import datetime

# Parameter Save Rate dalam detik, waktu mulai pengambilan frame, dan direktori penyimpanan
SR = 5  # Save Rate (misalnya, setiap 1/5 detik disimpan)
N = 5   # Waktu jeda sebelum mulai menyimpan frame (dalam detik)
DIR1 = 'saved_frames'  # Direktori tempat penyimpanan gambar

# Membuat direktori jika belum ada
if not os.path.exists(DIR1):
    os.makedirs(DIR1)

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

        # Menampilkan frame di jendela 'Webcam'
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Menghitung waktu berlalu sejak penyimpanan dimulai
    elapsed_capture_time = (current_time - start_capture_time).total_seconds()

    # Menampilkan jumlah file yang disimpan pada frame setelah penyimpanan dimulai
    cv2.putText(frame, f"Files saved: {file_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Elapsed Time: {int(elapsed_capture_time)}s", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Mengecek apakah waktunya untuk menyimpan frame
    if (current_time - BefTime).total_seconds() > 1 / SR:
        # Mendapatkan timestamp saat ini
        timestamp = current_time.strftime('%y%m%d%H%M%S%f')[:-3]  # Format YYMMDDHHmmSSMS
        filename = os.path.join(DIR1, f"{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame disimpan: {filename}")

        # Menambah jumlah file yang disimpan
        file_count += 1

        # Memperbarui waktu sebelumnya
        BefTime = current_time

    # Menampilkan frame di jendela 'Webcam'
    cv2.imshow('Webcam', frame)

    # Menunggu input tombol, keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan resource dan menutup jendela
cap.release()
cv2.destroyAllWindows()
