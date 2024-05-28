import os
import json
import cv2
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("silatus/1k_Website_Screenshots_and_Metadata")

# Print a sample to inspect the structure
print(dataset['train'][0])

# Define paths
images_dir = "D:\\int1\\dataset\\images"
annotations_dir = "D:\\int1\\dataset\\annotations"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Extract images and annotations
for i, data in enumerate(dataset['train']):
    img_path = data.get('image_path')  # Using get to avoid KeyError
    metadata = data.get('metadata')

    if not img_path or not metadata:
        print(f"Skipping entry {i} due to missing 'image_path' or 'metadata'")
        continue

    # Read and save the image
    img = cv2.imread(img_path)
    img_name = f"image_{i}.jpg"
    cv2.imwrite(os.path.join(images_dir, img_name), img)

    # Convert metadata to YOLO format
    annotations = []
    for input_element in metadata.get('inputs', []):
        if 'location' in input_element and 'size' in input_element:
            x_center = (input_element['location']['x'] + input_element['size']['width'] / 2) / img.shape[1]
            y_center = (input_element['location']['y'] + input_element['size']['height'] / 2) / img.shape[0]
            width = input_element['size']['width'] / img.shape[1]
            height = input_element['size']['height'] / img.shape[0]
            class_id = 0  # Assuming a single class for simplicity

            annotation = f"{class_id} {x_center} {y_center} {width} {height}"
            annotations.append(annotation)

    # Save annotations to a text file
    annotation_file = os.path.join(annotations_dir, f"image_{i}.txt")
    with open(annotation_file, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')

print("Conversion to YOLO format completed.")
