"""
ur3_interface.py
----------------
UR3 controller using the 'urx' library (socket‑based, works on CB3).
Includes the collections.Iterable patch for Python 3.11+ compatibility.
"""

# ---- CRITICAL PATCH: Fix collections.Iterable for math3d ----
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

# ---- Now import the rest ----
import time
import cv2
import numpy as np
from typing import Optional, List, Dict
import urx

from interfaces import CameraInterface, ArmPublisher, ArmMotionError
from board import ROWS, COLS, EMPTY, PLAYER, ROBOT, create_empty_grid, clone_grid

import sorting_config
from config import (
    TOOL_TCP_OFFSET,
    T_CAM_TO_BASE,
    T_BOARD_TO_BASE,
    COLUMN_POSITIONS,
    GAME_ORIENTATION,
    HOME_JOINTS,
    CATCHER_DROP_POSE,
    CATCHER_PICKUP_POSE,
    SIDE_CLEARANCE_OFFSET_M,
)


# ======================================================================
# UR3 ARM CONTROLLER WITH SUCTION – urx version
# ======================================================================

class UR3SuctionArmPublisher(ArmPublisher):
    def __init__(self,
                 robot_ip: str = "192.168.40.27",
                 suction_pin: int = 1,
                 home_position: list = HOME_JOINTS,
                 drop_height: float = 0.05,
                 approach_height: float = 0.05,
                 pickup_approach_height: float = 0.05,
                 pickup_drop_height: float = 0.020,
                 speed: float = 0.125,          # reduced from 0.3
                 acceleration: float = 0.1,    # reduced from 0.3
                 slow_speed: float = 0.05,
                 slow_acceleration: float = 0.04,
                 pickup_positions: Dict[int, list] = None):
        """
        Args:
            robot_ip: IP address of the UR3.
            suction_pin: Digital output pin for suction.
            home_position: Joint angles for home (radians).
            drop_height: Z height for placing chip on board (base frame).
            approach_height: Z height for approaching board.
            pickup_approach_height: Z height for approaching chip (safe).
            pickup_drop_height: Z height for actually touching the chip.
            speed, acceleration: normal motion parameters.
            slow_speed, slow_acceleration: used instead of speed/acceleration
                whenever _move_to_pose(..., slow=True) is called -- for
                precision-critical legs like the final descent into a
                column entrance, matching the same slow/careful
                philosophy already used for the sorting task's placement
                step.
            pickup_positions: fallback fixed positions for chips (dict {chip: [x,y,z]}).
        """
        self.robot_ip = robot_ip
        self.suction_pin = suction_pin
        self.home_position = home_position
        self.drop_height = drop_height
        self.approach_height = approach_height
        self.pickup_approach_height = pickup_approach_height
        self.pickup_drop_height = pickup_drop_height
        self.speed = speed
        self.acceleration = acceleration
        self.slow_speed = slow_speed
        self.slow_acceleration = slow_acceleration

        # Fixed orientation for the suction cup (always pointing down)
        self.fixed_rxryrz = [2.905, 1.199, 0]

        # Fallback fixed pickup positions (used only for the game, not sorting)
        self.pickup_positions = pickup_positions or {
            PLAYER: [0.30, 0.20, 0.10],
            ROBOT:  [0.30, -0.20, 0.10],
        }

        # Connect to the robot using urx
        try:
            self.robot = urx.Robot(robot_ip)
            print(f"✅ Connected to UR3 at {robot_ip} via urx")
        except Exception as e:
            raise RuntimeError(f"urx connection failed: {e}")

        # Set tool TCP offset (suction tool)
        self.robot.set_tcp(TOOL_TCP_OFFSET)
        print(f"TCP offset set to {TOOL_TCP_OFFSET}")

        # Ensure suction is OFF
        self.suction_off()

        # Move to home
        self._go_home()

    # ---------- Motion commands ----------
    def _go_home(self):
        print("Moving to home...")
        self.robot.movej(self.home_position, acc=self.acceleration, vel=self.speed * 2)

    def _move_to_pose(self, pose, is_joint=False, slow=False):
        """
        Move to a Cartesian pose [x,y,z,rx,ry,rz] or joint pose.
        is_joint: if True, `pose` is a joint list; if False, Cartesian.
        slow: if True, uses self.slow_speed/self.slow_acceleration
            instead of the normal self.speed/self.acceleration --
            intended for precision-critical legs (e.g. the final
            descent into a column entrance), where accuracy matters
            more than speed.
        """
        vel = self.slow_speed if slow else self.speed
        acc = self.slow_acceleration if slow else self.acceleration
        try:
            if is_joint:
                self.robot.movej(pose, acc=acc, vel=vel)
            else:
                self.robot.movel(pose, acc=acc, vel=vel)
        except Exception as e:
            # If motion fails, try to go home to recover
            try:
                self._go_home()
            except:
                pass
            raise ArmMotionError(f"Motion to pose {pose} failed: {e}")

    # ---------- Board‑relative column poses ----------
    def _get_column_drop_pose(self, col, approach=False, orientation='sort'):
        """
        Get the drop pose for a column, transformed from board frame to base frame.
        col: column index (0-6)
        approach: if True, return a pose above the drop point (for safe approach)
        orientation: 'sort' (suction down) or 'game' (vertical insertion)
        """
        x_b, y_b, z_b = COLUMN_POSITIONS[col]
        point_board = np.array([x_b, y_b, z_b, 1.0])
        point_base = T_BOARD_TO_BASE @ point_board
        x, y, z = point_base[0], point_base[1], point_base[2]

        if approach:
            z += self.approach_height

        if orientation == 'game':
            rx, ry, rz = GAME_ORIENTATION
        else:
            rx, ry, rz = self.fixed_rxryrz

        return [x, y, z, rx, ry, rz]

    # ---------- Suction ----------
    def suction_on(self):
        print("[Suction] ON")
        self.robot.set_digital_out(self.suction_pin, True)
        time.sleep(2.0)  # allow vacuum to build

    def suction_off(self):
        print("[Suction] OFF")
        self.robot.set_digital_out(self.suction_pin, False)
        time.sleep(2.0)

    # ---------- Generic pick‑and‑place (for sorting) ----------
    def pick_and_place_to_pose(self, pickup_pose, place_pose, wrist_yaw = 0.0):
        """
        Pick from pickup_pose and place at place_pose.
        Both poses are [x,y,z,rx,ry,rz] in robot base frame.
        wrist_yaw: rotation about Z (joint 6) – useful for tool offset.
        """
        try:
            # 1. Approach pickup
            approach = pickup_pose.copy()
            approach[2] = approach[2] + sorting_config.SORT_APPROACH_HEIGHT
            approach[5] = wrist_yaw
            self._move_to_pose(approach)

            # 2. Lower to pickup
            pick = pickup_pose.copy()
            pick[1] += 0.003
            pick[2] = pick[2] - self.pickup_drop_height
            self._move_to_pose(pick)

            # 3. Suction ON
            self.suction_on()

            # 4. Lift
            self._move_to_pose(approach)

            # 5. Move to place approach
            place_approach = place_pose.copy()
            place_approach[2] = place_approach[2] + sorting_config.SORT_APPROACH_HEIGHT
            place_approach[5] = wrist_yaw
            self._move_to_pose(place_approach)

            # 6. Lower to place
            self._move_to_pose(place_pose)

            # 7. Suction OFF
            self.suction_off()

            # 8. Retract
            self._move_to_pose(place_approach)
            self._go_home()

        except Exception as e:
            try:
                self.suction_off()
                self._go_home()
            except:
                pass
            raise ArmMotionError(f"Pick-and-place failed: {e}")

    # ---------- Game-specific move via the catcher fixture ----------
    def play_column_move(self, chip_pickup_pose, col):
        """
        Full two-stage move for playing a chip into a column during a
        real game:
          1. Pick the chip from its camera-detected position
             (chip_pickup_pose) and drop it into the fixed catcher --
             rough placement is fine here, the catcher's shape does the
             real alignment work via gravity, so this leg uses normal
             (not slow) movement.
          2. Pick the chip AGAIN from the catcher's known, fixed settled
             position -- this pickup is precise/centered, since the
             catcher guarantees the chip ends up in the exact same spot
             every time, regardless of how the first drop landed.
          3. Move to the column's approach point, with the game
             orientation already set via _get_column_drop_pose.
          4. Slow, linear descent into the column entrance -- same
             precision technique already used for the sorting task's
             placement step.
          5. Retreat straight back up to the approach point.
          6. Move laterally to the side (config.SIDE_CLEARANCE_OFFSET_M)
             -- an explicit collision-clearance step before going home,
             in case retreating straight up alone isn't enough
             clearance from the board.
          7. Return home.

        chip_pickup_pose: [x, y, z, rx, ry, rz] in base frame, the
            chip's camera-detected position (e.g. from
            CombinedCamera.get_chip_positions()).
        col: column index (0-6).
        """
        approach_pose = self._get_column_drop_pose(col, approach=True, orientation='game')
        drop_pose = self._get_column_drop_pose(col, approach=False, orientation='game')

        try:
            # ---- Stage A: chip's detected position -> catcher. Rough
            # placement is fine here -- normal speed, no slow descent.
            pickup_approach = list(chip_pickup_pose)
            pickup_approach[2] += sorting_config.SORT_APPROACH_HEIGHT
            self._move_to_pose(pickup_approach)
            chip_pose = chip_pickup_pose.copy()
            chip_pose[1] += 0.003
            chip_pose[2] = chip_pose[2] - self.pickup_drop_height
            self._move_to_pose(chip_pose)
            self.suction_on()
            self._move_to_pose(pickup_approach)

            catcher_drop_approach = list(CATCHER_DROP_POSE)
            catcher_drop_approach[2] += sorting_config.SORT_APPROACH_HEIGHT
            self._move_to_pose(catcher_drop_approach)
            self._move_to_pose(CATCHER_DROP_POSE)
            self.suction_off()
            self._move_to_pose(catcher_drop_approach)

            # ---- Stage B: catcher's fixed, settled position -> column.
            # This pickup is precise/centered since the catcher already
            # did the alignment work.
            catcher_pickup_approach = list(CATCHER_PICKUP_POSE)
            catcher_pickup_approach[2] += sorting_config.SORT_APPROACH_HEIGHT
            self._move_to_pose(catcher_pickup_approach)
            self._move_to_pose(CATCHER_PICKUP_POSE)
            self.suction_on()
            self._move_to_pose(catcher_pickup_approach)

            # Side pose approach for clearance, game orientation already set.
            side_pose = list(approach_pose)
            side_pose[0] -= SIDE_CLEARANCE_OFFSET_M[0] * (col + 3)
            self._move_to_pose(side_pose)

            # Column approach, game orientation already set.
            self._move_to_pose(approach_pose)

            # Slow, linear descent into the column entrance.
            self._move_to_pose(drop_pose, slow=True)
            self.suction_off()

            # Retreat straight back up to the approach point.
            self._move_to_pose(approach_pose, slow=True)

            # Move to the side -- explicit collision clearance before home.
            self._move_to_pose(side_pose)

            self._go_home()

        except Exception as e:
            try:
                self.suction_off()
                self._go_home()
            except:
                pass
            raise ArmMotionError(f"play_column_move failed: {e}")

    # ---------- Game‑specific move (uses fixed pickup positions) ----------
    def publish_move(self, row: int, col: int, chip: int):
        """
        Used by the game manager - picks a chip from a fixed pickup position
        and drops it into the specified column with game orientation.
        """
        pickup_pose_xyz = self.pickup_positions.get(chip)
        if pickup_pose_xyz is None:
            raise ArmMotionError(f"No pickup position for chip {chip}")
        pickup_pose = pickup_pose_xyz + self.fixed_rxryrz

        drop_pose = self._get_column_drop_pose(col, approach=False, orientation='game')

        self.pick_and_place_to_pose(pickup_pose, drop_pose)

    # ---------- Calibration stubs ----------
    def calibrate_column_positions(self):
        print("Column calibration not implemented - use board calibration.")

    def calibrate_pickup_positions(self):
        print("Pickup calibration not implemented - use teach pendant.")

    # ---------- Cleanup ----------
    def __del__(self):
        if hasattr(self, 'robot'):
            try:
                self.robot.close()
            except:
                pass


# ======================================================================
# REALSENSE 3D CAMERA INTERFACE – with tuned HSV
# ======================================================================

class RealsenseCameraInterface(CameraInterface):
    """
    Uses Intel RealSense D435 to detect chips using HSV ranges tuned via the slider script.
    """

    def __init__(self,
                 camera_serial: str = '',
                 transform_matrix: Optional[np.ndarray] = None,
                 board_corners: list = None,
                 color_ranges: dict = None):
        """
        Args:
            camera_serial: if multiple cameras, specify serial.
            transform_matrix: 4x4 transform from camera frame to robot base frame.
            board_corners: optional pre-calibrated corners for board detection.
            color_ranges: HSV ranges for red/yellow classification.
        """
        import pyrealsense2 as rs
        self.rs = rs

        # Setup pipeline
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        if camera_serial:
            config.enable_device(camera_serial)
        config.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.bgr8, 30)
        config.enable_stream(self.rs.stream.depth, 640, 480, self.rs.format.z16, 30)
        self.pipeline.start(config)

        # Align depth to color
        self.align = self.rs.align(self.rs.stream.color)

        # Get intrinsics
        profile = self.pipeline.get_active_profile()
        self.color_profile = profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        self.depth_profile = profile.get_stream(self.rs.stream.depth).as_video_stream_profile()
        self.intrinsics = self.color_profile.get_intrinsics()

        # Board detection state (optional)
        self.board_corners = board_corners
        self.calibrated = board_corners is not None

        # ---- TUNED HSV RANGES FROM test_chip_detection.py ----
        self.color_ranges = color_ranges or {
            'red_lower1': np.array([0, 80, 65]),
            'red_upper1': np.array([19, 255, 255]),
            'red_lower2': np.array([170, 50, 50]),
            'red_upper2': np.array([179, 255, 255]),
            'yellow_lower': np.array([20, 60, 50]),
            'yellow_upper': np.array([75, 255, 255]),
        }

        # Transform from camera to robot base
        self.T_cam_to_base = transform_matrix if transform_matrix is not None else np.eye(4)

        # Cache last grid
        self._last_grid = create_empty_grid()
        self._last_chip_positions = []

    def read_grid(self) -> list:
        """Capture RGB frame, detect board state, return 6x7 grid."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return clone_grid(self._last_grid)

        color_image = np.asanyarray(color_frame.get_data())
        grid = self._detect_board_state(color_image)
        if grid is not None:
            self._last_grid = grid
        return clone_grid(self._last_grid)

    def get_chip_positions(self, chip_type: int = None, min_area: int = 30) -> List[dict]:
        """
        Detect chips (red/yellow) on the table using the tuned HSV ranges.
        Returns a list of dicts: {'x':, 'y':, 'z':, 'color': 'red'/'yellow'}
        Coordinates are in robot base frame.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return []

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # ---- Red masks ----
        mask_red1 = cv2.inRange(hsv, self.color_ranges['red_lower1'], self.color_ranges['red_upper1'])
        mask_red2 = cv2.inRange(hsv, self.color_ranges['red_lower2'], self.color_ranges['red_upper2'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # ---- Yellow mask ----
        mask_yellow = cv2.inRange(hsv, self.color_ranges['yellow_lower'], self.color_ranges['yellow_upper'])

        chips = []

        # Process red contours
        red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            depth = depth_image[cy, cx] / 1000.0 * 0.985  # mm → m
            if depth == 0 or depth > 1.5:
                continue
            point_cam = self.rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth)
            pt_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            pt_base = self.T_cam_to_base @ pt_h
            chips.append({
                'x': pt_base[0],
                'y': pt_base[1],
                'z': pt_base[2],
                'color': 'red'
            })

        # Process yellow contours
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            depth = depth_image[cy, cx] / 1000.0 * 0.985
            if depth == 0 or depth > 1.5:
                continue
            point_cam = self.rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth)
            pt_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            pt_base = self.T_cam_to_base @ pt_h
            chips.append({
                'x': pt_base[0],
                'y': pt_base[1],
                'z': pt_base[2],
                'color': 'yellow'
            })

        return chips

    def _detect_board_state(self, color_image) -> Optional[list]:
        """Placeholder for board state detection – not used in sorting."""
        if not self.calibrated or self.board_corners is None:
            return None
        return clone_grid(self._last_grid)

    def calibrate_board(self, color_image=None):
        print("Board calibration not implemented – use board_calibration.py.")

    def set_transform(self, transform_matrix):
        self.T_cam_to_base = transform_matrix

    def __del__(self):
        try:
            self.pipeline.stop()
        except:
            pass


# ======================================================================
# SIMULATED 3D CAMERA (for testing)
# ======================================================================

class Simulated3DCameraInterface(CameraInterface):
    def __init__(self):
        self._grid = create_empty_grid()
        self._chip_positions = []
    def read_grid(self):
        return clone_grid(self._grid)
    def get_chip_positions(self, chip_type=None):
        return [{'x': 0.3, 'y': 0.1, 'z': 0.02, 'color': 'red'},
                {'x': 0.35, 'y': -0.1, 'z': 0.02, 'color': 'yellow'}]
    def set_grid(self, grid):
        self._grid = clone_grid(grid)
    def reset(self):
        self._grid = create_empty_grid()