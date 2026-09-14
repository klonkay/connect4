"""
simulation/robodk_camera.py
A drop-in replacement for a real camera (ur3_interface.py's
RealsenseCameraInterface / Simulated3DCameraInterface), backed by RoboDK
ground truth instead of actual image processing.

Grid reading works by matching each currently-known chip's position
against a set of predefined 'Cell_R<row>_C<col>' Frame items -- one per
board cell (rows 0-5, cols 0-6, matching board.py's ROWS/COLS and its
"row 0 = top" convention). Whichever cell frame a chip is closest to
(within CELL_MATCH_TOL_M) is read as occupied by that chip's color.

This intentionally does NOT depend on config.T_BOARD_TO_BASE or any
board-corner-rectangle math -- each cell frame already encodes exactly
where a chip resting in that slot should be, so no separate board
calibration step is needed for grid reading specifically.
"""

import numpy as np

from interfaces import CameraInterface
from board import PLAYER, ROBOT, ROWS, COLS, create_empty_grid, clone_grid


class RoboDKCellCamera(CameraInterface):
    CELL_MATCH_TOL_M = 0.003   # how close a chip must be to a cell frame to count as "in" it

    def __init__(self, rdk_iface, player_color, robot_color):
        """
        rdk_iface: an already-connected simulation.robodk_interface.RoboDKInterface
                   (share the SAME instance used by your arm publisher --
                   don't construct a second one, it would open a redundant
                   connection and redo TCP/collision setup unnecessarily).
        player_color / robot_color: the two chip color strings, matching
                   config.CHIP_ONE_COLOR / CHIP_TWO_COLOR, telling this
                   class which color maps to board.PLAYER vs board.ROBOT.
        """
        self.rdk_iface = rdk_iface
        self.player_color = player_color
        self.robot_color = robot_color
        self.cell_positions = self._load_cell_frames()

    def _load_cell_frames(self):
        """Looks up every Cell_R<row>_C<col> item once at construction
        time and caches its position in the robot base frame. Missing
        cells are warned about but don't crash -- they'll just never be
        detected as occupied."""
        positions = {}
        missing = []
        base_inv = self.rdk_iface._robot_base_pose_abs.inv()

        for row in range(ROWS):
            for col in range(COLS):
                name = f"Cell_R{row}_C{col}"
                item = self.rdk_iface.rdk.Item(name)
                if not item.Valid():
                    missing.append(name)
                    continue
                pose_in_base = base_inv * item.PoseAbs()
                transform = self.rdk_iface._mat_to_transform_m(pose_in_base)
                positions[(row, col)] = transform[:3, 3]

        if missing:
            preview = missing[:5]
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            print(f"Warning: {len(missing)} cell frame(s) not found in the station: "
                  f"{preview}{more}. Those cells will never be detected as occupied.")
        return positions

    def read_grid(self):
        """Scans all currently known chips and assigns each to whichever
        cell frame it's closest to, if within tolerance. Chips not
        resting near any cell (e.g. still at a column-entrance drop
        point, waiting to be manually dragged down) are simply not
        counted yet -- exactly the behavior needed for the "drag to
        simulate gravity, then game_manager notices the change" flow."""
        grid = create_empty_grid()

        for chip in self.rdk_iface.list_chip_items():
            chip_pos = chip["position"]
            best_cell, best_dist = None, float("inf")
            for (row, col), cell_pos in self.cell_positions.items():
                d = np.linalg.norm(chip_pos - cell_pos)
                if d < best_dist:
                    best_dist = d
                    best_cell = (row, col)
            if best_cell is not None and best_dist < self.CELL_MATCH_TOL_M:
                row, col = best_cell
                grid[row][col] = PLAYER if chip["color"] == self.player_color else ROBOT

        return grid

    def get_chip_positions(self, chip_type=None):
        """Returns [{'x','y','z','color'}, ...] for every currently known
        chip -- matches the dict format used by ur3_interface.py's own
        camera classes (RealsenseCameraInterface / Simulated3DCameraInterface)."""
        results = []
        for chip in self.rdk_iface.list_chip_items():
            x, y, z = chip["position"]
            results.append({"x": float(x), "y": float(y), "z": float(z), "color": chip["color"]})

        if chip_type is not None:
            target_color = self.player_color if chip_type == PLAYER else self.robot_color
            results = [c for c in results if c["color"] == target_color]
        return results

    def get_column_entrance_position(self, col, clearance_m=0.05, top_row=0):
        """Returns (x, y, z) in robot base frame for where a chip should
        be released to enter this column from above -- derived directly
        from the SAME Cell_R<row>_C<col> frame used for grid reading (the
        top row's cell), raised by `clearance_m`. This ties
        column-entrance geometry to the single source of truth already
        used for grid detection, instead of maintaining a second,
        independently hardcoded set of coordinates (like
        config.COLUMN_POSITIONS) that can silently go stale the moment
        the board or robot are repositioned."""
        pos = self.cell_positions.get((top_row, col))
        if pos is None:
            raise ValueError(f"No cell frame found for column {col} (row {top_row}); "
                              f"check that Cell_R{top_row}_C{col} exists in the station.")
        x, y, z = pos
        return x, y, z + clearance_m

    def reset(self):
        """Nothing to reset in simulation -- board state is always read
        live from ground truth. Present only so game_manager.py's
        hasattr(self.camera, 'reset') check finds it and skips the
        'clear the physical board' prompt."""
        pass
