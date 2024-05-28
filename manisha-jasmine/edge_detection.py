import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Load the image in color and convert to grayscale
image_path = 'image5.png'
image_color = cv2.imread(image_path)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Sobel operator for edge detection
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
sobel = np.uint8(sobel)

# Apply Canny edge detector
canny_edges = cv2.Canny(blurred_image, 100, 200)

# Apply OCR using Pytesseract
custom_config = r'--oem 3 --psm 6'  # Adjust OCR settings as needed
text_boxes = pytesseract.image_to_data(blurred_image, config=custom_config, output_type=pytesseract.Output.DICT)

# Display the original and edge-detected images
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Sobel Edge Detection')
plt.imshow(sobel, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Canny Edge Detection')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

# Draw bounding boxes around detected text
for i, (x, y, w, h) in enumerate(zip(text_boxes['left'], text_boxes['top'], text_boxes['width'], text_boxes['height'])):
    if text_boxes['text'][i].strip():  # Filter out empty strings
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Text Detection with Bounding Boxes')
plt.axis('off')

plt.tight_layout()
plt.show()
