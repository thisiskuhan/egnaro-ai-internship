import cv2
import numpy as np

def draw_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on their size and aspect ratio
    min_area = 1000
    max_aspect_ratio = 5
    
    input_fields = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        if cv2.contourArea(contour) > min_area and aspect_ratio < max_aspect_ratio:
            input_fields.append((x, y, w, h))
    
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