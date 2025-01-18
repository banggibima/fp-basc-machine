import os
import cv2
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocess import preprocess_image
from unet_model import build_unet
from utils import load_coco_annotations

# Function to convert bbox to mask
def bbox_to_mask(image_size, bbox):
    mask = np.zeros(image_size, dtype=np.uint8)
    x, y, w, h = map(lambda val: int(round(val)), bbox)  # Convert bbox coordinates to integers (rounded)
    mask[y:y+h, x:x+w] = 1
    return mask

# Function to convert polygon segmentation to mask
def polygon_to_mask(image_size, segmentation):
    mask = np.zeros(image_size, dtype=np.uint8)
    # Convert segmentation points from float to int
    points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [points], 1)
    return mask

def data_generator(annotations, image_dir, batch_size=16, image_size=(256, 256)):
    while True:
        batch_images = []
        batch_masks = []
        
        for image_info in annotations.values():
            # Get the full image path
            image_path = image_info["image_path"]
            # print(f"Processing image: {image_path}")
            
            # Load and preprocess image
            image = preprocess_image(image_path, output_size=image_size)
            
            # Initialize mask as an empty array
            mask = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)
            
            for ann in image_info["annotations"]:
                # Get the bounding box and segmentation
                bbox = ann["bbox"]
                segmentation = ann["segmentation"]

                # Debugging: Cek apakah segmentation dan bbox ada
                # print(f"Processing annotation: bbox: {bbox}, segmentation: {segmentation}")
                
                # Convert bbox or segmentation to mask
                if len(segmentation) > 0:
                    # Use polygon segmentation
                    mask = polygon_to_mask(mask.shape, segmentation[0])
                else:
                    # Use bounding box
                    mask = bbox_to_mask(mask.shape, bbox)

            # Append image and mask to the batch
            batch_images.append(image)
            batch_masks.append(np.expand_dims(mask, axis=-1))
            
            # Yield batch when batch size is met
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_masks)
                batch_images = []
                batch_masks = []

# Training function
def train_unet(train_dir, val_dir, output_model_path, batch_size=16):
    # Load annotations
    train_annotations = load_coco_annotations(os.path.join(train_dir, "_annotations.coco.json"), train_dir)
    val_annotations = load_coco_annotations(os.path.join(val_dir, "_annotations.coco.json"), val_dir)
    
    # Create data generators for training and validation
    train_gen = data_generator(train_annotations, train_dir, batch_size=batch_size)
    val_gen = data_generator(val_annotations, val_dir, batch_size=batch_size)
    
    # Build U-Net model
    model = build_unet()
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    
    # Setup checkpoints to save best model
    checkpoint = ModelCheckpoint(output_model_path, monitor="val_loss", save_best_only=True, verbose=1)
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_annotations) // batch_size,  # Adjust batch size
        validation_steps=len(val_annotations) // batch_size,  # Adjust batch size
        epochs=20,
        callbacks=[checkpoint]
    )
    
    return history

if __name__ == "__main__":
    train_dir = "datasets/train"  # Path to training data
    val_dir = "datasets/valid"   # Path to validation data
    output_model_path = "models/unet_license_plate_model.h5"  # Path to save the trained model

    # Call the train_unet function
    history = train_unet(train_dir, val_dir, output_model_path)
