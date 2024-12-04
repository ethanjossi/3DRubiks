import cv2
import numpy as np
import matplotlib.pyplot as plt

def rubiks_cube_line_detection_and_homography_clean(image):
    """
    Detect the entire Rubik's Cube face using line detection and correct its perspective.
    """
    # Create a figure for debugging outputs
    plt.figure(figsize=(12, 8))

    # Step 1: Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(4, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")

    # Step 2: Line Detection
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    debug_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(debug_lines, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines")
    plt.axis("off")

    # Step 3: Extract Bounding Box
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        points = np.array(points)

        # Find the convex hull of the points
        hull = cv2.convexHull(points)
        rect = cv2.boundingRect(hull)
        x, y, w, h = rect

        debug_bbox = image.copy()
        cv2.rectangle(debug_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.subplot(4, 4, 3)
        plt.imshow(cv2.cvtColor(debug_bbox, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box")
        plt.axis("off")
    else:
        raise ValueError("No lines detected.")

    # Step 4: Warp the Detected Face to a Square
    src_pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (300, 300))

    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Warped Square")
    plt.axis("off")

    # Step 5: Divide Warped Square into 3x3 Grid
    squares = []
    h, w = warped.shape[:2]
    step_h, step_w = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            square = warped[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            squares.append(square)

    # Visualize Extracted Squares (in the remaining subplots)
    for idx, square in enumerate(squares[:9]):  # Ensure we plot only 9 squares
        plt.subplot(4, 4, 5 + idx)
        plt.imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        plt.title(f"Square {idx + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return warped, squares







# Example usage for debugging
if __name__ == "__main__":
    # Load an example image
    image_file = "images/red2.jpeg"  # Replace with the path to your image
    image = cv2.imread(image_file)

    try:
        warped = rubiks_cube_line_detection_and_homography_clean(image)
        print("Preprocessing completed successfully!")
    except ValueError as e:
        print(f"Error: {e}")
