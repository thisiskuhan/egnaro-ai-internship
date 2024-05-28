import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import re
import numpy as np

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def draw_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use Tesseract to do OCR on the image
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Define regular expressions for username, email, phone, and password fields
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
            if re.search(email_phone_regex, text, re.IGNORECASE):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                username_boxes.append((x, y, w, h))
            elif re.search(password_regex, text, re.IGNORECASE):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                password_boxes.append((x, y, w, h))
    
    # Find bounding boxes around potential input fields
    potential_fields = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Ignore small contours
            potential_fields.append((x, y, w, h))
    
    # Match detected keywords with potential input fields
    def find_nearest_field(boxes, potential_fields):
        matched_boxes = []
        for (x, y, w, h) in boxes:
            min_dist = float('inf')
            nearest_field = None
            for (fx, fy, fw, fh) in potential_fields:
                dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                if dist < min_dist and dist < 50:  # Proximity threshold
                    min_dist = dist
                    nearest_field = (fx, fy, fw, fh)
            if nearest_field:
                matched_boxes.append(nearest_field)
        return matched_boxes
    
    matched_username_boxes = find_nearest_field(username_boxes, potential_fields)
    matched_password_boxes = find_nearest_field(password_boxes, potential_fields)
    
    # Draw bounding boxes around detected fields with padding
    padding = 10
    for (x, y, w, h) in matched_username_boxes:
        cv2.rectangle(image, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
    for (x, y, w, h) in matched_password_boxes:
        cv2.rectangle(image, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 0, 255), 2)
    
    # Save or display the output image
    output_image_path = 'output_image_with_boxes.jpg'
    cv2.imwrite(output_image_path, image)
    Image.open(output_image_path).show()
    
    # Print coordinates of username and password fields
    print(f"Detected {len(matched_username_boxes)} email/phone field(s) and {len(matched_password_boxes)} password field(s).")
    print("Email/Phone field coordinates (x, y, width, height):", matched_username_boxes)
    print("Password field coordinates (x, y, width, height):", matched_password_boxes)
    
    return matched_username_boxes, matched_password_boxes

# Call the function with the path to your image

username_coords, password_coords = draw_bounding_boxes('D:\\int1\\netflix.png')
