import cv2
import numpy as np
import pytesseract
import json
import os

def process_image(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def segment_plate(image, model):
    prediction = model.predict(image)[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

def perform_ocr(segmented_plate):
    text = pytesseract.image_to_string(segmented_plate, config="--psm 7")
    return text.strip()

def load_coco_annotations(annotation_file, image_dir):
    """
    Load COCO annotations and images information from a given annotation JSON file.
    
    :param annotation_file: Path to the annotation JSON file.
    :param image_dir: Directory containing the images.
    :return: A dictionary containing image and annotation data.
    """
    with open(annotation_file) as f:
        data = json.load(f)
    
    annotations_dict = {}
    
    # Create a mapping of category id to category name
    categories = {category['id']: category['name'] for category in data['categories']}
    
    for image_info in data['images']:
        # Get the image path
        image_path = os.path.join(image_dir, image_info['file_name'])
        
        # Collect annotations related to the current image
        image_annotations = []
        
        for annotation in data['annotations']:
            if annotation['image_id'] == image_info['id']:
                # Add annotation data (bbox, segmentation, category name)
                category_name = categories.get(annotation['category_id'], "unknown")
                image_annotations.append({
                    'bbox': annotation['bbox'],
                    'segmentation': annotation['segmentation'],
                    'category': category_name  # Add category name to annotation
                })
        
        # Store image info along with annotations in a dictionary
        annotations_dict[image_info['id']] = {
            'image_path': image_path,
            'annotations': image_annotations
        }
    
    return annotations_dict

