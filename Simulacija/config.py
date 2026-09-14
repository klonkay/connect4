"""
config.py
==========
Single source of truth for every measurable/tunable value in the system.
No other file should hard-code a dimension, color range, or robot address.

Fill in the TOOL_* section once you've physically measured your suction
tool; everything else (path planning, safety heights, TCP registration
with the robot controller) automatically adapts.
"""

import numpy as np

# ============================================================
# ROBOT CONNECTION
# ============================================================
ROBOT_IP = "192.168.40.27"          # UR3 controller IP address

# Motion tuning
DEFAULT_JOINT_SPEED = 1.05          # rad/s
DEFAULT_JOINT_ACCEL = 1.4           # rad/s^2
DEFAULT_LIN_SPEED = 0.25            # m/s  -- fast travel moves
DEFAULT_LIN_ACCEL = 0.5             # m/s^2
APPROACH_LIN_SPEED = 0.08           # m/s  -- slow moves near contact
APPROACH_LIN_ACCEL = 0.2            # m/s^2

# Resting / home joint configuration (radians). Teach this on the real
# robot (a pose clear of the camera FOV and the sorting surface) and
# paste the joint values here.
HOME_JOINTS = [0.0, -2.6180, 1.0472, -3.1416, 1.5708, -2.3562]
home_joints = [0, -150, 60, -180, 90.0, -45.0]

# ============================================================
# TOOL (SUCTION CUP) DIMENSIONS -- MEASURE AND FILL IN
# ============================================================
# All distances in meters. Defined relative to the robot's tool0 flange
# frame, along the tool's own Z axis (pointing out of the cup) unless
# stated otherwise.

TOOL_FLANGE_TO_CUP_TIP = 0.1     # flange face -> cup sealing plane at rest length
TOOL_CUP_RADIUS = 0.005            # outer radius of the cup's sealing contact patch
TOOL_CUP_COMPRESSION = 0.002       # how far the cup compresses vertically when it
                                    # seals against a flat surface (affects standoff)

# Active TCP sent to the robot controller: [x, y, z, rx, ry, rz].
# Rotation stays at 0 unless the cup is mounted off the flange's central axis.
TOOL_TCP_OFFSET = [0.0375, -0.0365, TOOL_FLANGE_TO_CUP_TIP, 0.0, 0.0, 0.0]

# Standoffs derived from the tool, used everywhere in path planning.
TOOL_APPROACH_CLEARANCE = 0.05     # m above chip/placement surface where the
                                    # arm switches from fast travel to slow approach
TOOL_SAFE_TRAVEL_HEIGHT = 0.15     # m minimum height above the working surface
                                    # used for all lateral (non pick/place) travel

# ============================================================
# CHIP GEOMETRY (sanity checks / grasp validation)
# ============================================================
CHIP_DIAMETER_RANGE = (0.020, 0.045)     # m, (min, max) accepted chip diameter
CHIP_MIN_FLAT_AREA_M2 = 6.0e-4            # m^2, minimum face area accepted as a chip

# ============================================================
# CHIP COLORS -- edit to match your actual chip set
# ============================================================
# HSV is far more lighting-robust than RGB for this kind of segmentation.
# Ranges are OpenCV convention: H in [0,179], S,V in [0,255].
# If H_min > H_max the range is treated as wrapping around 0/179 (e.g. red).

CHIP_ONE_COLOR = "red"
CHIP_ONE_HSV_RANGE = ((170, 120, 70), (10, 255, 255))     # wraps around 0

CHIP_TWO_COLOR = "yellow"
CHIP_TWO_HSV_RANGE = ((20, 100, 100), (35, 255, 255))

# Off-color border ring found on the smaller face of each chip. Only used
# to tell which physical face is currently visible (informational/logging);
# it does not affect which face suction is applied to.
CHIP_BORDER_HSV_RANGE = ((0, 0, 0), (179, 60, 60))   # placeholder: dark/neutral

# ============================================================
# BOARD PLAYING SURFACE -- BOARD-LOCAL frame, meters
# ============================================================
# Two independent rectangles, one per half -- measured directly on the
# physical board relative to BoardFrame, never in the robot's base frame.
# They don't need to share an exact common edge ("roughly" adjoining is
# fine); each is just where that half's chips are allowed to sit.
#
# Example values below match: robot's half y in (220, 440)mm, player's
# half y in (0, 220)mm, both spanning x in (0, 300)mm -- EDIT to your
# actual measured board.
BOARD_ROBOT_HALF_CORNER_A = [-0.220, 0.005 ]    # (x, y), board-local
BOARD_ROBOT_HALF_CORNER_B = [-0.005, 0.230]    # (x, y), board-local
BOARD_PLAYER_HALF_CORNER_A = [0.005, 0.005]   # (x, y), board-local
BOARD_PLAYER_HALF_CORNER_B = [0.220, 0.230]   # (x, y), board-local
SURFACE_Z_BOARD = 0.000   # board-local height of the playing surface (both halves)

# Spacing between placed chips within a half (must exceed chip diameter).
PLACEMENT_GRID_PITCH = 0.03   # m

# Flat thickness of a single chip, used to compute the height of a second
# chip stacked directly on top of an existing one when the destination
# half has no remaining collision-free flat space -- see
# planning/placement_manager.py's find_placement().
CHIP_THICKNESS_M = 0.004   # m -- MEASURE YOUR REAL CHIPS AND SET

# ============================================================
# BOARD COLLISION AVOIDANCE
# ============================================================
# How high (above SURFACE_Z_BOARD) the board's physical structure could
# stick up -- raised walls, dividers, edge trim, etc.
BOARD_OBSTRUCTION_HEIGHT = 0.010   # m -- MEASURE YOUR BOARD'S TALLEST FEATURE AND SET

# Extra clearance added on top of BOARD_OBSTRUCTION_HEIGHT for LATERAL
# travel specifically. This exists because our path planning only
# reasons about the TOOL TIP's position -- it has no model of the rest
# of the arm's body (elbow, forearm, wrist), which can be significantly
# wider than the tool and end up clipping the board even when the TCP
# path itself is clear, especially in elbow-out IK configurations. This
# margin is a heuristic, not a guarantee -- it does not replace real
# whole-arm collision checking (which is what RoboDK's own collision
# engine, enabled in simulation/robodk_interface.py, is actually for).
# If you still see arm-vs-board collisions with a generous margin here,
# the real fix is more likely a different robot mounting position/angle
# relative to the board, not a bigger number.
ROBOT_BODY_CLEARANCE_MARGIN = 0.01   # m -- generous default, tune to your robot's actual link widths

# ============================================================
# GAME LOGIC
# ============================================================
# At the start of a game (a "reset"), chips can be scattered across BOTH
# halves in any mix -- not necessarily cleanly separated by color. The
# player picks a color at startup; the robot then moves:
#   - every chip of the PLAYER's color currently on the robot's half -> player's half
#   - every chip of the OTHER color currently on the player's half   -> robot's half
# Chips already on their correct half are left untouched (never even
# selected as targets) and count as already sorted. See
# task/sorting_task.py's _pending_targets() / planning/placement_manager.py's
# half_of_point().

# ============================================================
# BOARD CALIBRATION
# ============================================================
# A single ArUco marker fixed to the physical board defines the board's
# own coordinate frame; the corners above are expressed relative to it.
# Run board_calibration.py whenever the board is placed or moved, and
# paste the resulting matrix below.
BOARD_MARKER_DICT = "DICT_5X5_50"   # must match the marker you printed
BOARD_MARKER_ID = 0                  # the specific marker ID used on the board
BOARD_MARKER_LENGTH_M = 0.05         # side length of the printed marker, meters

# Transform from the board's local frame (== the marker's frame) to the
# robot base frame. Placeholder = identity; board_calibration.py computes
# the real value using the camera's T_CAM_TO_BASE above.
T_BOARD_TO_BASE = np.eye(4)


# ============================================================
# CAMERA
# ============================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Hand-eye calibration result: transform from camera optical frame to
# robot base frame, as a 4x4 homogeneous matrix. Placeholder = identity.
# Run calibration.py to generate the real value, then paste it here.
T_CAM_TO_BASE = np.eye(4)

# ============================================================
# VISION / DETECTION TUNING
# ============================================================
DEPTH_MIN_VALID = 0.05      # m, ignore depth readings closer than this
DEPTH_MAX_VALID = 1.50      # m, ignore depth readings farther than this
MORPH_KERNEL_SIZE = 5       # px, mask cleanup kernel

# ============================================================
# TASK / MISC
# ============================================================
POLL_PERIOD_S = 0.2         # seconds between detection loop iterations
MAX_SORT_RETRIES = 3         # per-chip retry count on a failed pick
