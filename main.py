import rubiks as rb
import singleImageProcess as ip
import cv2

""" front is red middle
    back is orange middle
    left is green middle
    right is blue middle
    top is white middle
    bottom is yellow middle
""" 
side_map = {
    'red': 'front',
    'orange': 'back',
    'green': 'left',
    'blue': 'right',
    'white': 'top',
    'yellow': 'bottom'
}

image_paths = ["images/redPerfect.jpeg",
               "images/orangePerfect.jpg", 
               "images/yellowPerfect.jpg", 
               "images/whitePerfect.jpg",
               "images/greenPerfect.jpg",
               "images/bluePerfect.jpg"]
# Instantiate a rubik's cube
cube = rb.RubiksCube()
# Read in the image
for image_path in image_paths:
    image = cv2.imread(image_path)
    # Get the 9 squares
    warped, squares = ip.rubiks_cube_line_detection_and_homography_clean(image)
    kmeans_results = []
    color_name = []
    for square in squares:
        kmeans_color = ip.detect_dominant_color_kmeans(square)
        kmeans_results.append(kmeans_color)
    print(kmeans_results)
    color_matrix = [
        [ip.get_color_name(rgb) for rgb in row]
        for row in [kmeans_results[i:i+3] for i in range(0, len(kmeans_results), 3)]
    ]
    print(color_matrix)
    cube.set_side(side=side_map[color_matrix[1][1]],    colors=color_matrix)
cube.display_rubiks_cube()