import os
import cv2
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocess import preprocess_image
from unet_model import build_unet
from utils import load_coco_annotations

# Fungsi untuk mengonversi bounding box menjadi mask
def bbox_to_mask(image_size, bbox):
    mask = np.zeros(image_size, dtype=np.uint8)
    x, y, w, h = map(lambda val: int(round(val)), bbox)  # Konversi koordinat bbox ke integer
    mask[y:y+h, x:x+w] = 1
    return mask

# Fungsi untuk mengonversi polygon segmentation menjadi mask
def polygon_to_mask(image_size, segmentation):
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)  # Konversi titik ke integer
    cv2.fillPoly(mask, [points], 1)  # Gambar polygon pada mask
    return mask

# Generator data untuk menghasilkan batch gambar dan mask
def data_generator(annotations, image_dir, batch_size=16, image_size=(256, 256)):
    while True:
        batch_images = []
        batch_masks = []
        
        for image_info in annotations.values():
            image_path = image_info["image_path"]  # Path gambar
            image = preprocess_image(image_path, output_size=image_size)  # Preprocessing gambar
            mask = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)  # Inisialisasi mask
            
            for ann in image_info["annotations"]:
                bbox = ann["bbox"]
                segmentation = ann["segmentation"]
                
                if len(segmentation) > 0:  # Gunakan segmentation jika tersedia
                    mask = polygon_to_mask(mask.shape, segmentation[0])
                else:  # Jika tidak, gunakan bbox
                    mask = bbox_to_mask(mask.shape, bbox)

            batch_images.append(image)
            batch_masks.append(np.expand_dims(mask, axis=-1))
            
            if len(batch_images) == batch_size:  # Jika batch penuh, kembalikan batch
                yield np.array(batch_images), np.array(batch_masks)
                batch_images = []
                batch_masks = []

# Fungsi untuk melatih model U-Net
def train_unet(train_dir, val_dir, output_model_path, batch_size=16):
    train_annotations = load_coco_annotations(os.path.join(train_dir, "_annotations.coco.json"), train_dir)  # Load anotasi pelatihan
    val_annotations = load_coco_annotations(os.path.join(val_dir, "_annotations.coco.json"), val_dir)  # Load anotasi validasi
    
    train_gen = data_generator(train_annotations, train_dir, batch_size=batch_size)  # Generator data pelatihan
    val_gen = data_generator(val_annotations, val_dir, batch_size=batch_size)  # Generator data validasi
    
    model = build_unet()  # Bangun model U-Net
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])  # Compile model
    
    checkpoint = ModelCheckpoint(output_model_path, monitor="val_loss", save_best_only=True, verbose=1)  # Simpan model terbaik
    
    # Melatih model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_annotations) // batch_size,
        validation_steps=len(val_annotations) // batch_size,
        epochs=20,
        callbacks=[checkpoint]
    )
    
    return history

if __name__ == "__main__":
    train_dir = "datasets/train"  # Direktori data pelatihan
    val_dir = "datasets/valid"   # Direktori data validasi
    output_model_path = "models/unet_license_plate_model.h5"  # Path untuk menyimpan model

    history = train_unet(train_dir, val_dir, output_model_path)  # Panggil fungsi pelatihan
