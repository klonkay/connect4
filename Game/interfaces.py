"""
interfaces.py
-------------
This module defines abstract interfaces for camera and arm control,
and provides mock implementations for testing without hardware.
"""

import time
from abc import ABC, abstractmethod

from board import ROWS, COLS, EMPTY, PLAYER, ROBOT, create_empty_grid, clone_grid


class CameraInterface(ABC):
    @abstractmethod
    def read_grid(self):
        """Return the latest known 6x7 grid state observed by the camera."""
        raise NotImplementedError


class ArmPublisher(ABC):
    @abstractmethod
    def publish_move(self, row, col, chip):
        """Publish the target cell (row, col) for the arm to drop `chip` into."""
        raise NotImplementedError


class ArmMotionError(Exception):
    """
    Raised by an ArmPublisher implementation when a commanded move fails.
    """
    pass


class MockCameraInterface(CameraInterface):
    """
    Simulates the camera by holding an in-memory grid.
    """
    def __init__(self):
        self._grid = create_empty_grid()

    def read_grid(self):
        return clone_grid(self._grid)

    def force_set_grid(self, grid):
        self._grid = clone_grid(grid)

    def reset(self):
        self._grid = create_empty_grid()


class MockArmPublisher(ArmPublisher):
    """
    Simulates the arm by writing directly into the mock camera's grid.
    """
    def __init__(self, camera: MockCameraInterface, simulated_move_seconds=0.5):
        self._camera = camera
        self._simulated_move_seconds = simulated_move_seconds

    def publish_move(self, row, col, chip):
        print(f"[MockArm] Command received: drop chip {chip} at (row={row}, col={col}). "
              f"Simulating arm motion...")
        time.sleep(self._simulated_move_seconds)
        grid = self._camera.read_grid()
        grid[row][col] = chip
        self._camera.force_set_grid(grid)
        print("[MockArm] Move complete, new grid state should now be visible to camera.")