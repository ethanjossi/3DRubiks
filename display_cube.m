% Define Rubik's Cube using a struct
rubiksCube.front = repmat("red", 3, 3);
rubiksCube.back = repmat("orange", 3, 3);
rubiksCube.left = repmat("green", 3, 3);
rubiksCube.right = repmat("blue", 3, 3);
rubiksCube.top = repmat("white", 3, 3);
rubiksCube.bottom = repmat("yellow", 3, 3);

% Display the Rubik's Cube
displayRubiksCube(rubiksCube);

function displayRubiksCube(cube)
    % Create a figure for the cube
    figure;
    hold on;
    axis equal;
    axis off;
    
    % Define face positions
    facePositions = {
        'front', [0, 0, 1.33333];
        'back', [0, 0, -1.33333];
        'left', [-1.33333, 0, 0];
        'right', [1.33333, 0, 0];
        'top', [0, 1.33333, 0];
        'bottom', [0, -1.33333, 0];
    };
    
    % Define face orientations
    faceRotations = {
        'front', eye(3);
        'back', diag([-1, 1, -1]);
        'left', [0 0 1; 0 1 0; -1 0 0];
        'right', [0 0 -1; 0 1 0; 1 0 0];
        'top', [1 0 0; 0 0 -1; 0 1 0];
        'bottom', [1 0 0; 0 0 1; 0 -1 0];
    };
    
    % Define colors
    colorMap = struct('red', [1, 0, 0], 'orange', [1, 0.5, 0], ...
                      'green', [0, 1, 0], 'blue', [0, 0, 1], ...
                      'white', [1, 1, 1], 'yellow', [1, 1, 0]);
    
    % Draw each face
    for i = 1:size(facePositions, 1)
        face = facePositions{i, 1};
        center = facePositions{i, 2};
        rotation = faceRotations{i, 2};
        
        % Get the colors for the current face
        colors = cube.(face);
        
        % Draw the face
        drawFace(center, rotation, colors, colorMap);
    end
end

function drawFace(center, rotation, colors, colorMap)
    % Size of one tile
    tileSize = 0.9;
    
    % Loop through the 3x3 grid of the face
    for row = 1:3
        for col = 1:3
            % Compute tile center in face-local coordinates
            x = (col - 2) * tileSize;
            y = (2 - row) * tileSize;
            z = 0;
            localCenter = [x, y, z];
            
            % Transform to world coordinates
            worldCenter = center + (rotation * localCenter')';
            
            % Get the color of the tile
            tileColor = colors(row, col);
            if isfield(colorMap, tileColor)
                faceColor = colorMap.(tileColor);
            else
                faceColor = [0, 0, 0]; % Default to black for unknown colors
            end
            
            % Draw the tile as a patch (square)
            drawTile(worldCenter, rotation, tileSize, faceColor);
        end
    end
end

function drawTile(center, rotation, tileSize, faceColor)
    % Define a square in the XY plane
    halfSize = tileSize / 2;
    square = [-halfSize, -halfSize, 0;
               halfSize, -halfSize, 0;
               halfSize,  halfSize, 0;
              -halfSize,  halfSize, 0];
    
    % Rotate the square to align with the face
    rotatedSquare = (rotation * square')';
    
    % Translate to the correct position
    translatedSquare = rotatedSquare + center;
    
    % Draw the square as a patch
    patch('Vertices', translatedSquare, ...
          'Faces', [1, 2, 3, 4], ...
          'FaceColor', faceColor, ...
          'EdgeColor', 'k', ...
          'LineWidth', 1);
end
