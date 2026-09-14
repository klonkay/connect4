"""
sorting_config.py
------------------
Configuration for chip sorting.
Edit the corner coordinates after calibration.
"""

# Board halves in board-local frame (x, y, z) – z is the surface height.
# These are rectangles defined by two opposite corners.
SORT_PLAYER_HALF_CORNERS = [
    [0.006, -0.127, 0.0],   # (x1, y1, z) in board frame
    [0.211, 0.068, 0.0]    # (x2, y2, z)
]
SORT_ROBOT_HALF_CORNERS = [
    [-0.199, -0.127, 0.0],
    [0.006, 0.068, 0.0]
]

# Placement parameters
PLACEMENT_GRID_PITCH = 0.035        # m between chip centres
CHIP_RADIUS = 0.0135                 # m – adjust to your chip size
STACK_OFFSET_Z = 0.005              # m – vertical offset per stacked layer

# Approach heights (m) relative to the table surface
SORT_APPROACH_HEIGHT = 0.05
SORT_TRAVEL_HEIGHT = 0.05

# Retry counts
MAX_RETRIES_PER_CHIP = 3