import os
import cv2
import numpy as np

# Fungsi untuk memproses gambar: membaca, mengubah ukuran, dan normalisasi
def preprocess_image(image_path, output_size=(256, 256)):
    image = cv2.imread(image_path)  # Membaca gambar dari file
    image = cv2.resize(image, output_size)  # Mengubah ukuran gambar
    return image / 255.0  # Normalisasi ke rentang [0, 1]

# Fungsi untuk memproses seluruh dataset secara batch
def preprocess_dataset(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Membuat direktori output jika belum ada
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)  # Path gambar asli
        preprocessed_image = preprocess_image(image_path)  # Memproses gambar
        output_path = os.path.join(output_dir, image_name)  # Path untuk menyimpan hasil
        cv2.imwrite(output_path, (preprocessed_image * 255).astype(np.uint8))  # Menyimpan hasil
