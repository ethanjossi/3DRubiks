import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image_debug_find_face(image):
    """
    Preprocess the input image to detect the outer quadrilateral of the Rubik's Cube face.
    Includes debugging visualizations for intermediate steps.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 5)
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    # Step 1: Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    plt.subplot(2, 3, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")

    # Step 2: Contour Detection
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_contours = image.copy()
    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(debug_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contours Detected")
    plt.axis("off")

    # Step 3: Filter for Largest Contours
    contour_areas = [cv2.contourArea(c) for c in contours]
    largest_contours = sorted(zip(contours, contour_areas), key=lambda x: x[1], reverse=True)

    # Try to find a quadrilateral from the largest few contours
    approx = None
    for contour, area in largest_contours[:5]:  # Analyze only the top 5 largest contours
        epsilon = 0.02 * cv2.arcLength(contour, True)
        candidate_approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(candidate_approx) == 4:  # Found a quadrilateral
            approx = candidate_approx
            break

    if approx is None:
        print("DEBUG: Could not find a quadrilateral for the Rubik's Cube face.")
        raise ValueError("Could not detect the Rubik's Cube face.")

    # Debug visualization for the chosen quadrilateral
    debug_approx = image.copy()
    cv2.drawContours(debug_approx, [approx], -1, (255, 0, 0), 2)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(debug_approx, cv2.COLOR_BGR2RGB))
    plt.title("Quadrilateral Approximation")
    plt.axis("off")

    # Step 4: Homography (Warp to a Square)
    pts_src = np.array([point[0] for point in approx], dtype="float32")
    size = 300  # Output square size (300x300 pixels)
    pts_dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Warped Square")
    plt.axis("off")

    # Display the image processing pipeline
    plt.tight_layout()
    plt.show()

    return warped

# Example usage for debugging
if __name__ == "__main__":
    # Load an example image
    image_file = "red2.jpeg"  # Replace with the path to your image
    image = cv2.imread(image_file)

    try:
        warped = preprocess_image_debug_find_face(image)
        print("Preprocessing completed successfully!")
    except ValueError as e:
        print(f"Error: {e}")
