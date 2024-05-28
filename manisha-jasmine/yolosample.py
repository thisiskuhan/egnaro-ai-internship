import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
import numpy as np

# Load the custom YOLOv8 model
model = YOLO('D:/best (1).pt')

# Function to generate a color map
def generate_color_map(num_classes):
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
    return colors

# Function to perform inference, visualize the results, and print coordinates
def detect_and_visualize(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image)

    # Extract bounding boxes, confidence scores, and class labels
    boxes = torch.tensor(results[0].boxes.xyxy).cpu()  # Bounding boxes
    scores = torch.tensor(results[0].boxes.conf).cpu()  # Confidence scores
    class_ids = torch.tensor(results[0].boxes.cls).cpu()  # Class IDs

    # Apply Non-Maximum Suppression (NMS)
    iou_threshold = 0.5  # Intersection over Union threshold for NMS
    nms_indices = nms(boxes, scores, iou_threshold)

    # Filter boxes, scores, and class_ids using NMS indices
    boxes = boxes[nms_indices].numpy()
    scores = scores[nms_indices].numpy()
    class_ids = class_ids[nms_indices].numpy()

    # Generate color map
    num_classes = len(model.names)
    colors = generate_color_map(num_classes)

    # Loop through the detections, draw them on the image, and print coordinates
    for box, score, class_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = map(int, box)
        label = f'{model.names[int(class_id)]}: {score:.2f}'

        # Get color for the current class
        color = [int(c) for c in colors[int(class_id)]]

        # Draw bounding box and label on the image
        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Print coordinates and other information
        print(f"Component: {model.names[int(class_id)]}, Confidence: {score:.2f}")
        print(f"Coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    # Display the image with detections using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

# Path to your image
image_path = 'yolo5.png'

# Perform detection and visualize results
detect_and_visualize(image_path, model)
