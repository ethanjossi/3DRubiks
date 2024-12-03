import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    """
    Preprocess the input image: enhance edges, adjust brightness, and warp to a square.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150)
    print(edges)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    print(approx)

    # Ensure the contour is a quadrilateral
    if len(approx) != 4:
        raise ValueError("Could not find a quadrilateral for the Rubik's Cube face.")

    # Get the points of the quadrilateral
    pts_src = np.array([point[0] for point in approx], dtype="float32")

    # Define the points for the destination (perfect square)
    size = 300  # Output square size (300x300 pixels)
    pts_dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")

    # Compute the homography matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Warp the image to a square
    warped = cv2.warpPerspective(image, matrix, (size, size))
    return warped

def extract_colors(image):
    """
    Extract the dominant color from each 3x3 grid cell of the Rubik's Cube face.
    """
    colors = []
    size = image.shape[0]  # Assuming square image
    cell_size = size // 3  # Size of each grid cell

    # Convert to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for row in range(3):
        row_colors = []
        for col in range(3):
            # Extract the cell region
            cell = hsv_image[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size]

            # Compute the dominant color in HSV
            avg_color = cv2.mean(cell)[:3]
            row_colors.append(determine_color(avg_color))
        colors.append(row_colors)
    return colors

def determine_color(hsv_color):
    """
    Determine the Rubik's Cube color based on the HSV value.
    """
    h, s, v = hsv_color
    if s < 50 and v > 200:
        return "white"
    elif v < 50:
        return "black"
    elif 20 < h < 30:
        return "yellow"
    elif 0 < h < 10 or 160 < h < 180:
        return "red"
    elif 30 < h < 90:
        return "green"
    elif 90 < h < 150:
        return "blue"
    else:
        return "orange"

def process_rubiks_cube(images):
    """
    Process 6 images of the Rubik's Cube to extract colors and display results.
    """
    cube = {}
    face_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    processed_images = []

    for i, image in enumerate(images):
        try:
            # Preprocess the image and warp to a square
            warped = preprocess_image(image)
            processed_images.append(warped)

            # Extract colors and store in the cube data structure
            cube[face_names[i]] = extract_colors(warped)
        except ValueError as e:
            print(f"Error processing face {face_names[i]}: {e}")
            exit(1)  # Exit the program if an error occurs

    # Display the processed images
    plt.figure(figsize=(10, 6))
    for i, img in enumerate(processed_images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(face_names[i])
        plt.axis('off')
    plt.show()

    return cube

# Example usage
if __name__ == "__main__":
    # Load six images (replace with your own image file paths)
    image_files = ["blue.jpeg", "green.jpeg", "orange.jpeg", "red.jpeg", "white.jpeg", "yellow.jpeg"]
    images = [cv2.imread(file) for file in image_files]

    # Process the Rubik's Cube and get the color data
    rubiks_cube = process_rubiks_cube(images)
    print("Rubik's Cube Data Structure:")
    print(rubiks_cube)
