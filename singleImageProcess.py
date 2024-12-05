# Importing necessary libraries again
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Reimplementing the required functions from the script
def rubiks_cube_line_detection_and_homography_clean(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        points = np.array(points)
        hull = cv2.convexHull(points)
        rect = cv2.boundingRect(hull)
        x, y, w, h = rect
        src_pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (300, 300))
        squares = []
        h, w = warped.shape[:2]
        step_h, step_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                square = warped[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
                squares.append(square)
        return warped, squares
    else:
        raise ValueError("No lines detected.")

def detect_dominant_color_kmeans(square):
    # Reshape the square into a list of pixels
    pixels = square.reshape(-1, 3)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    kmeans.fit(pixels)
    
    # Retrieve cluster labels and their counts
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # Retrieve cluster centers
    cluster_centers = kmeans.cluster_centers_
    
  
    
    # Filter out clusters that are too dark (close to black)
    brightness_threshold = 50  # Adjust as needed (lower values are stricter)
    valid_clusters = [
        (center, count) for center, count in zip(cluster_centers, counts)
        if np.mean(center) > brightness_threshold  # Mean brightness threshold
    ]
    
    # Debugging: Print valid clusters
    
    # If no valid clusters remain, return black as a fallback
    if not valid_clusters:
        print("No valid clusters found; returning [0, 0, 0].")
        return np.array([0, 0, 0])  # Black
    
    # Find the most frequent valid cluster
    dominant_cluster = max(valid_clusters, key=lambda x: x[1])  # Cluster with the highest count
    dominant_color = dominant_cluster[0].astype(int)

    
    return dominant_color
def normalize_illumination(square):
    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv_normalized = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

# Color name mapping
COLOR_NAME_REF = {
    "red": [40, 80, 210],
    "blue": [100, 45, 20],
    "green": [60, 155, 70],
    "yellow": [50, 135, 200],
    "orange": [30, 100, 220],
    "white": [155, 180, 200],
    
}

def get_color_name(rgb):
    """
    Find the closest color name for a given RGB value.
    """
    closest_color = None
    min_distance = float('inf')
    for color_name, ref_rgb in COLOR_NAME_REF.items():
        dist = distance.euclidean(rgb, ref_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_color = color_name
    return closest_color

def display_results_with_names(squares, results, title):
    """
    Display squares with their detected dominant colors and names.
    """
    plt.figure(figsize=(12, 8))
    for i, (square, rgb) in enumerate(zip(squares, results)):
        color_name = get_color_name(rgb)
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        plt.title(f"{color_name} (RGB: {rgb})")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Load the uploaded image
image_path = "images/redTilted.jpeg"
image = cv2.imread(image_path)

# Process the image
warped, squares = rubiks_cube_line_detection_and_homography_clean(image)

# Perform K-Means and Illumination Normalization
kmeans_results = []
illumination_results = []
normalized_squares = []

for square in squares:
    kmeans_color = detect_dominant_color_kmeans(square)
    kmeans_results.append(kmeans_color)
    # normalized_square = normalize_illumination(square)
    # normalized_squares.append(normalized_square)
    # normalized_kmeans_color = detect_dominant_color_kmeans(normalized_square)
    # illumination_results.append(normalized_kmeans_color)

# Display results with color names
# display_results_with_names(squares, kmeans_results, "K-Means Detected Colors with Names")
# display_results_with_names(normalized_squares, illumination_results, "Illumination-Normalized Colors with Names")
