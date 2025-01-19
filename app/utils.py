import cv2
import numpy as np
import pytesseract
import json
import os

# Fungsi untuk memproses gambar input
def process_image(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)  # Konversi bytes ke array NumPy
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode gambar
    image = cv2.resize(image, (256, 256))  # Ubah ukuran gambar menjadi 256x256
    image = image / 255.0  # Normalisasi gambar ke rentang [0, 1]
    return np.expand_dims(image, axis=0)  # Tambahkan dimensi batch

# Fungsi untuk segmentasi plat nomor menggunakan model
def segment_plate(image, model):
    prediction = model.predict(image)[0, :, :, 0]  # Prediksi segmentasi
    prediction = (prediction > 0.5).astype(np.uint8) * 255  # Thresholding ke biner
    return prediction

# Fungsi untuk melakukan OCR pada hasil segmentasi
def perform_ocr(segmented_plate):
    text = pytesseract.image_to_string(segmented_plate, config="--psm 7")  # Ekstraksi teks
    return text.strip()  # Hapus spasi di awal/akhir teks

# Fungsi untuk memuat anotasi COCO
def load_coco_annotations(annotation_file, image_dir):
    with open(annotation_file) as f:
        data = json.load(f)  # Baca file anotasi
    
    annotations_dict = {}
    categories = {category['id']: category['name'] for category in data['categories']}  # Peta id ke nama kategori
    
    for image_info in data['images']:
        image_path = os.path.join(image_dir, image_info['file_name'])  # Path gambar
        image_annotations = []
        
        for annotation in data['annotations']:
            if annotation['image_id'] == image_info['id']:
                # Tambahkan data anotasi
                category_name = categories.get(annotation['category_id'], "unknown")
                image_annotations.append({
                    'bbox': annotation['bbox'],
                    'segmentation': annotation['segmentation'],
                    'category': category_name
                })
        
        annotations_dict[image_info['id']] = {
            'image_path': image_path,
            'annotations': image_annotations
        }
    
    return annotations_dict
