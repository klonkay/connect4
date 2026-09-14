"""
main.py
Entry point: wires the camera, robot controller, and sorting task
together and starts the sort loop.

Before running:
  1. Fill in config.ROBOT_IP and config.HOME_JOINTS.
  2. Measure and fill in the TOOL_* dimensions in config.py.
  3. Run calibration.py once and paste the result into config.T_CAM_TO_BASE.
  4. Confirm config.CHIP_ONE_COLOR / CHIP_TWO_COLOR and their HSV ranges
     match your actual chips (a quick script that prints HSV under your
     lighting is the fastest way to tune these).
  5. Measure config.BOARD_ROBOT_HALF_CORNER_A/B, config.BOARD_PLAYER_HALF_CORNER_A/B,
     and config.SURFACE_Z_BOARD for your physical board, relative to your
     BoardFrame marker.
"""

from robot.robot_controller import RobotController
from vision.camera_interface import CameraInterface
from task.sorting_task import SortingTask
import config


def _prompt_for_color():
    options = (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR)
    while True:
        choice = input(f"Which color is the player using? {options}: ").strip().lower()
        if choice in options:
            return choice
        print(f"'{choice}' isn't one of {options}, try again.")


def main():
    player_color = _prompt_for_color()
    camera = CameraInterface()
    robot = RobotController()
    task = SortingTask(camera, robot, player_color)
    try:
        task.run()
    finally:
        camera.stop()
        robot.shutdown()


if __name__ == "__main__":
    main()
