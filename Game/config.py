"""
config.py
==========
Global configuration for the UR3 Connect4 project.
Fill in your robot IP, home joint angles, and colour names.
Transforms (T_CAM_TO_BASE, T_BOARD_TO_BASE) will be computed by calibration scripts.
"""

import math
import numpy as np

ROBOT_IP = "192.168.40.27"          # Your UR3's IP address
HOME_JOINTS = [3.141592, -2.00712864, -1.13446401, -1.57079633, 1.57079633, 0.78539816]   # safe home (radians)


CHIP_ONE_COLOR = "red"
CHIP_TWO_COLOR = "yellow"


# Transform from camera optical frame to robot base frame (4x4)
T_CAM_TO_BASE = np.array([
    [
      0.999408362248595,
      -0.0030775384906966685,
      -0.034255718127339185,
      0.16317309238064354
    ],
    [
      0.018840913137587546,
      -0.7842642226151594,
      0.6201408300684463,
      -0.1457341616253577
    ],
    [
      -0.02877404142144957,
      -0.6204193403518926,
      -0.7837422386586032,
      0.762062878713212
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  ])

# Transform from board frame (ArUco marker) to robot base frame (4x4)
T_BOARD_TO_BASE = np.array([
  [
    0.9999033153716275,
    -0.012773161075827176,
    -0.005496022649166396,
    0.1929451315166213
  ],
  [
    0.012723101459632733,
    0.999878112785166,
    -0.009048881832782822,
    0.3889446193524273
  ],
  [
    0.00561093557947963,
    0.008978080491215821,
    0.9999439541657406,
    0.04660731563593623
  ],
  [
    0.0,
    0.0,
    0.0,
    1.0
  ]
])

# Column positions in BOARD-LOCAL frame (x, y, z) in meters 
'''COLUMN_POSITIONS = [
    (-0.090, 0.267, 0.25),  # Column 0
    (-0.060, 0.267, 0.25),  # Column 1
    (-0.030, 0.267, 0.25),  # Column 2
    (0.0, 0.267, 0.25),  # Column 3
    (0.030, 0.267, 0.25),  # Column 4
    (0.060, 0.267, 0.25),  # Column 5
    (0.090, 0.267, 0.25)   # Column 6
]'''

COLUMN_POSITIONS = [
    (-0.086, 0.093, 0.255),  # Column 0
    (-0.056, 0.093, 0.255),  # Column 1
    (-0.026, 0.093, 0.255),  # Column 2
    (0.004, 0.093, 0.255),  # Column 3
    (0.034, 0.093, 0.255),  # Column 4
    (0.064, 0.093, 0.255),  # Column 5
    (0.094, 0.093, 0.255)   # Column 6
]

GAME_ORIENTATION = [3.930, 1.649, 1.640] 


# ============================================================
# SORTING / PLACEMENT (fallback values – not critical for testing)
# ============================================================
DEFAULT_JOINT_SPEED = 0.015
DEFAULT_JOINT_ACCEL = 0.8
DEFAULT_LIN_SPEED = 0.1
DEFAULT_LIN_ACCEL = 0.5
APPROACH_LIN_SPEED = 0.08
APPROACH_LIN_ACCEL = 0.2

TOOL_TCP_OFFSET = [0.032, -0.033, 0.1, 0, 0, 0]   # set if you have a tool offset

SORT_ROI = (480, 315, 960, 515)   # adjust to your sorting area


# ============================================================
# CATCHER FIXTURE (chip alignment before column insertion)
# ============================================================
# During a real game move, the chip is first picked from wherever the
# camera detected it and dropped roughly into a fixed, slanted catcher
# -- rough placement is fine here, since the catcher's shape settles
# the chip into an exact, repeatable position via gravity. The chip is
# then picked up AGAIN from that settled position for a precise,
# centered grip before being played into the column.
#
# PLACEHOLDERS -- both must be measured and updated once the catcher is
# physically mounted. Format: [x, y, z, rx, ry, rz] in robot base frame.
CATCHER_DROP_POSE = [-0.1109, 0.4164, 0.0905, 2.921, 1.264, -1.191]     # where the chip is released INTO the catcher
CATCHER_PICKUP_POSE = [-0.09474, 0.41770, 0.07190, 2.921, 1.264, -1.191]  # the catcher's fixed settled/pickup position
 
# Lateral clearance offset (dx, dy, dz in meters, base frame) applied
# after retreating from a column, before returning home -- an explicit
# step to make sure the arm has cleared the board's structure before
# attempting a joint-space move home.
# PLACEHOLDER -- tune the direction/magnitude to whatever actually
# clears your physical board once the catcher and columns are in use.
SIDE_CLEARANCE_OFFSET_M = [0.030, 0.0, 0.0]
