import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def find_input_fields(image, text_box):
    x, y, w, h = text_box
    padding = 10
    
    # Extract the region of interest (ROI) below the text
    roi = image[y + h + padding:y + h + padding + 50, x:x + w + 100]
    
    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges_roi = cv2.Canny(blurred_roi, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on their size and aspect ratio
    min_area = 500
    max_aspect_ratio = 5
    
    input_fields = []
    for contour in contours:
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(contour)
        aspect_ratio = w_roi / float(h_roi)
        
        if cv2.contourArea(contour) > min_area and aspect_ratio < max_aspect_ratio:
            input_fields.append((x + x_roi, y + h + padding + y_roi, w_roi, h_roi))
    
    return input_fields

def draw_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Use Tesseract to do OCR on the image
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Define regular expressions for email, phone, and password fields
    email_phone_regex = r'(?i)email|phone|mobile'
    password_regex = r'(?i)password'
    
    # Initialize lists to store coordinates of username and password fields
    username_boxes = []
    password_boxes = []
    
    # Iterate over each word detected by Tesseract
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence threshold
            text = d['text'][i]
            if text.lower() in ['email', 'phone', 'mobile']:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                username_boxes.append((x, y, w, h))
            elif text.lower() == 'password':
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                password_boxes.append((x, y, w, h))
    
    # Find input fields near the detected text labels
    input_fields = []
    for box in username_boxes + password_boxes:
        input_fields.extend(find_input_fields(image, box))
    
    # Draw bounding boxes around detected input fields with padding
    padding = 5
    for (x, y, w, h) in input_fields:
        cv2.rectangle(image, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
    
    # Save or display the output image
    output_image_path = 'output_image_with_boxes.jpg'
    cv2.imwrite(output_image_path, image)
    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print coordinates of input fields
    print(f"Detected {len(input_fields)} input field(s).")
    print("Input field coordinates (x, y, width, height):", input_fields)
    
    return input_fields

# Call the function with the path to your image
input_field_coords = draw_bounding_boxes('D:\\int1\\netflix.png')