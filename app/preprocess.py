import os
import cv2
import numpy as np

def preprocess_image(image_path, output_size=(256, 256)):
    image = cv2.imread(image_path)  # Read the image from the file
    image = cv2.resize(image, output_size)  # Resize the image to the desired output size
    return image / 255.0  # Normalize the image to [0, 1]

# Example usage for batch preprocessing:
def preprocess_dataset(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        preprocessed_image = preprocess_image(image_path)
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, (preprocessed_image * 255).astype(np.uint8))
