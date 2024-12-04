import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_and_adjust_rubiks_detection(image):
    """
    Improved detection of Rubik's Cube squares with dynamic rotation and alignment.
    """
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        raise ValueError("No lines detected, unable to proceed.")
    
    # Generate grid intersections
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.extend([(x1, y1), (x2, y2)])
    points = np.array(points)

    # Find convex hull of points and fit a polygon
    hull = cv2.convexHull(points)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Ensure the box aligns to a square and warp
    side_length = max(rect[1])  # Ensure square by using the larger dimension
    dst_pts = np.float32([[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]])
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    warped = cv2.warpPerspective(image, M, (int(side_length), int(side_length)))

    # Divide warped square into 3x3 grid
    squares = []
    h, w = warped.shape[:2]
    step_h, step_w = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            square = warped[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            squares.append(square)

    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.title("Edges Detected")
    plt.axis("off")

    plt.subplot(4, 4, 3)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Aligned Bounding Box")
    plt.axis("off")

    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Warped Square")
    plt.axis("off")
 

    # Visualize individual squares
    
    for idx, square in enumerate(squares[:9]):  # Ensure only 9 squares are shown
        plt.subplot(4, 4, idx + 4)
        plt.imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        plt.title(f"Square {idx + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return warped, squares





# Example usage for debugging
if __name__ == "__main__":
    # Load an example image
    image_file = "images/redTilted.jpeg"  # Replace with the path to your image
    image = cv2.imread(image_file)

    try:
        warped =  refine_and_adjust_rubiks_detection(image)
        print("Preprocessing completed successfully!")
    except ValueError as e:
        print(f"Error: {e}")
