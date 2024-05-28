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

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use edge detection to find contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find rectangles
    rects = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Look for 4-sided polygons
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.5 < aspect_ratio < 5 and 20 < w < 500 and 20 < h < 100:  # Typical aspect ratio and size range for input boxes
                rects.append((x, y, w, h))

    # Use Tesseract to do OCR on the image
    d = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Initialize lists to store coordinates of username and password fields
    username_boxes = []
    password_boxes = []

    # Iterate over each word detected by Tesseract
    n_boxes = len(d['text'])
    text_boxes = []
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence threshold
            text = d['text'][i].lower()
            (text_x, text_y, text_w, text_h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text_boxes.append((text, text_x, text_y, text_w, text_h))

    # Associate text labels with nearest input fields
    for text, text_x, text_y, text_w, text_h in text_boxes:
        if 'email' in text or 'phone' in text or 'username' in text:
            nearest_rect = min(rects, key=lambda rect: ((rect[0] - text_x) ** 2 + (rect[1] - text_y) ** 2))
            username_boxes.append(nearest_rect)
        elif 'password' in text:
            nearest_rect = min(rects, key=lambda rect: ((rect[0] - text_x) ** 2 + (rect[1] - text_y) ** 2))
            password_boxes.append(nearest_rect)

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

username_coords, password_coords = draw_bounding_boxes('D:\\int1\\fb.png')
