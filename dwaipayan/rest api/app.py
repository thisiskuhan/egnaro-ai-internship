import os
import cv2
import torch
from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='hbk.pt')

def convert_coordinates(image_shape, boxes):
    height, width = image_shape
    converted_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box['xcenter'], box['ycenter'], box['width'], box['height']
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        converted_boxes.append({
            'class': int(box['class']),
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })
    return converted_boxes

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    # Load the image
    image = Image.open(filepath)
    results = model(image)
    predictions = results.pandas().xywh[0].to_dict(orient='records')

    # Convert coordinates
    converted_boxes = convert_coordinates(image.size, predictions)

    os.remove(filepath)
    return jsonify({'boxes': converted_boxes})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
