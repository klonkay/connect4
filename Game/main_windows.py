"""
main_windows.py
----------------
Windows entry point for UR3 Connect 4 with 3D vision.
Supports:
  --simulate          : no hardware, terminal play
  --camera-3d         : use RealSense 3D camera (instead of webcam)
  --robot-ip IP       : IP of UR3
  --suction-pin N     : digital output pin for suction
  --calibrate         : run calibration (columns + pickup)
  --test              : run tests
  --sort              : run sorting task only
"""

import sys
import argparse
import cv2
import numpy as np

import board as board_mod
from ai import RobotAI
from game_manager import GameManager
from interfaces import MockCameraInterface, MockArmPublisher
from sorting_task import SortingTask
import config

# Import only what exists in ur3_interface.py
from ur3_interface import (
    UR3SuctionArmPublisher,
    ArmMotionError,
)
from combined_camera import CombinedCamera


# ---------- Simple WebCam fallback (2D camera) ----------
class WebCamCameraInterface:
    """
    Minimal webcam interface using OpenCV.
    """
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_grid(self):
        # Placeholder – real board detection can be added later.
        # For sorting, we don't actually use this method.
        return [[0]*7 for _ in range(6)]

    def get_chip_positions(self, min_area=30):
        """Return empty list because webcam cannot do 3D."""
        # If you want 2D detection, you could implement it here,
        # but for sorting we need 3D poses, so we'll just return [].
        return []

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


# ---------- User input helpers ----------
def console_player_move_fn(grid):
    """Simulation-only input."""
    while True:
        raw = input(f"Your move - enter column (0-{board_mod.COLS-1}): ").strip()
        try:
            col = int(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if col < 0 or col >= board_mod.COLS:
            print(f"Column must be 0..{board_mod.COLS-1}.")
            continue
        if board_mod.next_open_row(grid, col) is None:
            print(f"Column {col} full.")
            continue
        return col


# ---------- Sorting mode ----------
def run_sort_mode(robot_ip, suction_pin=1, camera_index=0):
    """Run sorting task only."""
    print("=== SORTING MODE ===")
    try:
        camera = CombinedCamera(chip_roi=config.SORT_ROI, transform_matrix=config.T_CAM_TO_BASE)
        print("Using CombinedCamera (chip ROI + grid detection).")
    except Exception as e:
        print(f"CombinedCamera init failed: {e}. Falling back to webcam.")
        try:
            camera = WebCamCameraInterface(camera_index)
            print("Using webcam (2D only - sorting will not work without 3D).")
        except Exception as e2:
            print(f"Webcam also failed: {e2}. Falling back to mock camera.")
            camera = MockCameraInterface()

    try:
        arm = UR3SuctionArmPublisher(robot_ip, suction_pin=suction_pin)
    except Exception as e:
        print(f"Robot init failed: {e}")
        return

    # Ask player colour
    options = (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR)
    while True:
        choice = input(f"Which color are you playing? {options}: ").strip().lower()
        if choice in options:
            player_color = choice
            break
        print(f"'{choice}' not valid.")

    task = SortingTask(camera, arm, player_color)
    task.run()

    # Optionally start game
    proceed = input("Sorting complete. Start game? [y/n]: ").strip().lower()
    if proceed.startswith('y'):
        from game_manager import GameManager
        if hasattr(camera, 'set_player_robot_colors'):
            camera.set_player_robot_colors(player_color, task.robot_color)
        ai = RobotAI(depth=6)
        manager = GameManager(camera, arm, ai, verbose=True,
                               player_color=player_color, robot_color=task.robot_color)
        manager.run()


# ---------- Simulation mode ----------
def run_simulated_mode(use_mock_arm=True):
    """Simulation (no hardware) – uses terminal input."""
    print("=== SIMULATION MODE ===")
    if use_mock_arm:
        camera = MockCameraInterface()
        arm = MockArmPublisher(camera)
    else:
        # Use a mock camera that can provide fake chip positions
        camera = MockCameraInterface()  # or a custom Simulated3DCameraInterface
        arm = MockArmPublisher(camera)
    ai = RobotAI(depth=5)
    manager = GameManager(
        camera=camera,
        arm_publisher=arm,
        ai=ai,
        poll_interval=0.1,
        robot_move_timeout=30.0,
        player_move_timeout=None,
        player_move_input_fn=console_player_move_fn,
        verbose=True,
    )
    manager.run()


# ---------- Real hardware mode ----------
def run_real_mode(robot_ip: str, use_3d_camera: bool = False,
                  camera_index: int = 0, calibrate: bool = False,
                  suction_pin: int = 1):
    """Real hardware mode."""
    print(f"=== REAL MODE (3D={'Yes' if use_3d_camera else 'No'}) ===")
    print(f"Robot IP: {robot_ip}, Suction pin: DO{suction_pin}")

    # ----- Camera -----
    if use_3d_camera:
        print("Initialising CombinedCamera...")
        try:
            camera = CombinedCamera(chip_roi=config.SORT_ROI, transform_matrix=config.T_CAM_TO_BASE)
            print("CombinedCamera ready.")
            if calibrate:
                print("Calibration not needed - use live_grid_detection.py to tune.")
        except Exception as e:
            print(f"CombinedCamera init failed: {e}")
            print("Falling back to webcam.")
            try:
                camera = WebCamCameraInterface(camera_index)
                print("Using webcam (2D).")
            except Exception as e2:
                print(f"Webcam failed: {e2}. Falling back to mock camera.")
                camera = MockCameraInterface()
    else:
        print("Using 2D webcam.")
        try:
            camera = WebCamCameraInterface(camera_index)
            print("Webcam ready.")
            if calibrate:
                print("Calibrate board corners not implemented for webcam yet.")
        except Exception as e:
            print(f"Webcam init failed: {e}")
            print("Falling back to mock camera.")
            camera = MockCameraInterface()

    # ----- Robot -----
    print("Initialising UR3 with suction...")
    try:
        arm = UR3SuctionArmPublisher(robot_ip, suction_pin=suction_pin)
        if calibrate:
            # These calibration methods need to be implemented in ur3_interface.py
            # If they don't exist, you can comment them out or add stubs.
            print("Calibrating column positions...")
            # arm.calibrate_column_positions()
            print("Calibrating pickup positions...")
            # arm.calibrate_pickup_positions()
            print("Calibration methods not implemented - skipping.")
    except Exception as e:
        print(f"Robot init failed: {e}")
        return

    # ----- Player color -----
    options = (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR)
    while True:
        choice = input(f"Which color are you playing? {options}: ").strip().lower()
        if choice in options:
            player_color = choice
            break
        print(f"'{choice}' not valid.")
    robot_color = config.CHIP_TWO_COLOR if player_color == config.CHIP_ONE_COLOR else config.CHIP_ONE_COLOR

    # Camera needs to know which color means PLAYER vs ROBOT for grid
    # reading -- it's constructed before this prompt happens, so this
    # has to be set here rather than at construction time.
    if hasattr(camera, 'set_player_robot_colors'):
        camera.set_player_robot_colors(player_color, robot_color)

    # ----- AI -----
    ai = RobotAI(depth=6)

    # ----- Game Manager -----
    manager = GameManager(
        camera=camera,
        arm_publisher=arm,
        ai=ai,
        poll_interval=1.0,           # check every second
        stable_checks=3,             # require 3 identical reads
        robot_settle_time=1.0,       # wait 2.5s after robot move
        player_color=player_color,
        robot_color=robot_color,
        verbose=True,
    )

    print("\n=== GAME READY ===")
    print("Press Ctrl+C to stop.")
    try:
        manager.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        try:
            arm.suction_off()
            arm._go_home()
        except:
            pass
        print("Goodbye!")


# ---------- Main entry ----------
def main():
    parser = argparse.ArgumentParser(description="UR3 Connect 4 with 3D vision")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--camera-3d", action="store_true", help="Use RealSense 3D camera")
    parser.add_argument("--robot-ip", default="192.168.40.27", help="UR3 IP address")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (if not 3D)")
    parser.add_argument("--suction-pin", type=int, default=1, help="Digital output for suction")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--sort", action="store_true", help="Run sorting task only.")

    args = parser.parse_args()

    if args.sort:
        run_sort_mode(args.robot_ip, args.suction_pin, args.camera_index)
        return

    if args.test:
        import test_foul_detection
        test_foul_detection.test_basic_win_detection()
        test_foul_detection.test_ai_blocks_immediate_loss()
        test_foul_detection.test_ai_takes_immediate_win()
        test_foul_detection.test_fog_of_war_no_floating_chips()
        test_foul_detection.test_multi_move_foul_detected()
        test_foul_detection.test_floating_chip_foul_detected()
        test_foul_detection.test_game_manager_end_to_end_with_foul()
        print("All tests passed.")
        return

    if args.simulate:
        run_simulated_mode()
    else:
        run_real_mode(
            robot_ip=args.robot_ip,
            use_3d_camera=args.camera_3d,
            camera_index=args.camera_index,
            calibrate=args.calibrate,
            suction_pin=args.suction_pin,
        )


if __name__ == "__main__":
    main()