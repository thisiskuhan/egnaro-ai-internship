import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Function to classify the detected component based on text content and font size
def classify_component(text, font_size):
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["submit", "button", "next"]):
        return "Button"
    elif font_size > 30:  # Adjust threshold as needed
        return "Heading"
    elif any(keyword in text_lower for keyword in ["password", "username", "input", "textarea", "email","enter"]):
        return "Text Label"
    elif any(keyword in text_lower for keyword in ["input", "textarea", "email"]):
        return "Text Box"
    elif any(keyword in text_lower for keyword in ["link", "url", "anchor", "forgot"]):
        return "Link"
    else:
        return "Text"  # Default classification

# Load the image in color and convert to grayscale
image_path = 'image5.png'
image_color = cv2.imread(image_path)

if image_color is None:
    print(f"Error: Unable to load image at path '{image_path}'. Please check the file path and try again.")
    exit()

image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Apply Canny edge detector
canny_edges = cv2.Canny(blurred_image, 100, 200)

# Apply OCR using Pytesseract
custom_config = r'--oem 3 --psm 11'  # Adjust OCR settings as needed
text_boxes = pytesseract.image_to_data(blurred_image, config=custom_config, output_type=pytesseract.Output.DICT)

# Draw bounding boxes around detected text and display component types and positions
for i, (x, y, w, h) in enumerate(zip(text_boxes['left'], text_boxes['top'], text_boxes['width'], text_boxes['height'])):
    if text_boxes['text'][i].strip():  # Filter out empty strings
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        component_text = text_boxes['text'][i]
        component_font_size = text_boxes['height'][i]
        component_type = classify_component(component_text, component_font_size)
        # Adjust text label position to avoid overlap with bounding boxes
        text_x = x
        text_y = y - 10  # Adjust vertical position
        # Display component type and position (x, y coordinates)
        label = f"{component_type} ({x}, {y})"
        # Create a white background for the text label to improve readability
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_color, (text_x, text_y - label_height - baseline), (text_x + label_width, text_y + baseline), (255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(image_color, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the image with bounding boxes and component information
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Text Detection with Component Classification and Positions')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
