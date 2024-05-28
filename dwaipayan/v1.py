import cv2
import pytesseract
from pytesseract import Output
from PIL import Image

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

def draw_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Use Tesseract to do OCR on the image
    d = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Initialize lists to store coordinates of username and password fields
    username_boxes = []
    password_boxes = []

    # Iterate over each word detected by Tesseract
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence threshold
            text = d['text'][i].lower()
            if 'email' in text or 'phone' in text:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                username_boxes.append((x, y, w, h))
            elif 'password' in text:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                password_boxes.append((x, y, w, h))

    # Draw bounding boxes around detected fields
    for (x, y, w, h) in username_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in password_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save or display the output image
    output_image_path = 'output_image_with_boxes.jpg'
    cv2.imwrite(output_image_path, image)
    Image.open(output_image_path).show()

    # Print coordinates of username and password fields
    print("Username field coordinates (x, y, width, height):", username_boxes)
    print("Password field coordinates (x, y, width, height):", password_boxes)

    return username_boxes, password_boxes

# Call the function with the path to your image
username_coords, password_coords = draw_bounding_boxes('D:\\int1\\netflix.png')
