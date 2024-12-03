import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def display_rubiks_cube(cube):
    """
    Display a 3D representation of the Rubik's Cube.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Keep cube proportions

    # Define face positions and rotations
    face_positions = {
        'front': np.array([0, 0, 1.33333]),
        'back': np.array([0, 0, -1.33333]),
        'left': np.array([-1.33333, 0, 0]),
        'right': np.array([1.33333, 0, 0]),
        'top': np.array([0, 1.33333, 0]),
        'bottom': np.array([0, -1.33333, 0])
    }

    face_rotations = {
        'front': np.eye(3),
        'back': np.diag([-1, 1, -1]),
        'left': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        'right': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        'top': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        'bottom': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    }

    # Color map for face colors
    color_map = {
        'red': 'red',
        'orange': 'orange',
        'green': 'green',
        'blue': 'blue',
        'white': 'white',
        'yellow': 'yellow'
    }

    # Draw each face
    for face, position in face_positions.items():
        rotation = face_rotations[face]
        colors = cube[face]
        draw_face(ax, position, rotation, colors, color_map)

    # Adjust the view
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.axis('off')
    plt.show()

def draw_face(ax, center, rotation, colors, color_map):
    """
    Draw a single face of the Rubik's Cube.
    """
    tile_size = 0.9
    for row in range(3):
        for col in range(3):
            # Local coordinates of the tile
            x = (col - 1) * tile_size
            y = (1 - row) * tile_size
            z = 0
            local_center = np.array([x, y, z])
            world_center = center + rotation @ local_center

            # Tile color
            tile_color = color_map.get(colors[row][col], 'black')

            # Draw the tile
            draw_tile(ax, world_center, rotation, tile_size, tile_color)

def draw_tile(ax, center, rotation, tile_size, color):
    """
    Draw a single tile as a square on the cube face.
    """
    half_size = tile_size / 2
    square = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ])
    rotated_square = (rotation @ square.T).T + center
    face = Poly3DCollection([rotated_square], color=color, edgecolor='k', linewidth=1)
    ax.add_collection3d(face)

# Define the Rubik's Cube
rubiks_cube = {
    'front': [['red', 'red', 'red'], ['red', 'red', 'red'], ['red', 'red', 'red']],
    'back': [['orange', 'orange', 'orange'], ['orange', 'orange', 'orange'], ['orange', 'orange', 'orange']],
    'left': [['green', 'green', 'green'], ['green', 'green', 'green'], ['green', 'green', 'green']],
    'right': [['blue', 'blue', 'blue'], ['blue', 'blue', 'blue'], ['blue', 'blue', 'blue']],
    'top': [['white', 'white', 'white'], ['white', 'white', 'white'], ['white', 'white', 'white']],
    'bottom': [['yellow', 'yellow', 'yellow'], ['yellow', 'yellow', 'yellow'], ['yellow', 'yellow', 'yellow']]
}

# Display the Rubik's Cube
display_rubiks_cube(rubiks_cube)
