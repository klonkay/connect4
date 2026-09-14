"""
planning/placement_manager.py
Decides WHERE to place a chip on its destination half, based on the
board's ACTUAL current occupancy -- not an abstract in-memory grid. This
matters once the board can be completely full (e.g. all 42 chips
present): a pre-computed grid has no idea which slots are already taken
by chips that existed before this run started, so it needs to check real
positions instead.

Placement logic, in order:
    1. Find collision-free empty space on the destination half -> place
       there, flat on the board surface.
    2. If no empty space exists anywhere in that half, fall back to
       stacking directly on top of an existing SAME-COLOR chip there.
    3. If neither is possible, return None -- the caller should try a
       different chip instead of forcing an impossible placement.

All geometry is computed in the BOARD's own local frame (so it stays
correct regardless of the board's rotation on the table) and only
transformed into the robot base frame at the point a specific placement
is actually returned.
"""

import numpy as np
import config
from planning.geometry_utils import transform_point, transform_vector


class PlacementManager:
    def __init__(self):
        self.half_bounds_board = self._compute_half_bounds_board()
        self._spiral_offsets = self._build_spiral_offsets(max_ring=25)
        self.surface_normal_base = self._compute_surface_normal()
        self._T_base_to_board = np.linalg.inv(config.T_BOARD_TO_BASE)

    def _compute_surface_normal(self):
        normal = transform_vector(config.T_BOARD_TO_BASE, [0.0, 0.0, 1.0])
        return normal / np.linalg.norm(normal)

    @staticmethod
    def _rect(corner_a, corner_b):
        xa, ya = corner_a
        xb, yb = corner_b
        x_min, x_max = min(xa, xb), max(xa, xb)
        y_min, y_max = min(ya, yb), max(ya, yb)
        return {
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
            "center": np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]),
        }

    def _compute_half_bounds_board(self):
        return {
            "robot": self._rect(config.BOARD_ROBOT_HALF_CORNER_A,
                                 config.BOARD_ROBOT_HALF_CORNER_B),
            "player": self._rect(config.BOARD_PLAYER_HALF_CORNER_A,
                                  config.BOARD_PLAYER_HALF_CORNER_B),
        }

    @staticmethod
    def _build_spiral_offsets(max_ring):
        """Integer (i, j) grid offsets ordered by increasing distance from
        the origin, so offset 0 is dead center and later ones ring
        outward. Used purely to search candidate positions in a sensible
        center-out order -- NOT as a record of what's occupied, since
        that's now checked against real chip positions instead."""
        offsets = [(0, 0)]
        for ring in range(1, max_ring + 1):
            for i in range(-ring, ring + 1):
                for j in range(-ring, ring + 1):
                    if max(abs(i), abs(j)) == ring:
                        offsets.append((i, j))
        offsets.sort(key=lambda ij: ij[0] ** 2 + ij[1] ** 2)
        return offsets

    # ---------- half classification ----------
    def opposite_half(self, half_key):
        return "player" if half_key == "robot" else "robot"

    def half_of_point(self, xy_base):
        """Classifies a detected chip position (robot base frame, [x, y])
        into 'robot' or 'player' by checking which half-rectangle it
        falls inside, in board-local coordinates. Falls back to whichever
        half's center is closer if the point lands in neither rectangle."""
        point_board = self._base_xy_to_board_xy(xy_base)

        for half_key, half in self.half_bounds_board.items():
            x_lo, x_hi = half["x_range"]
            y_lo, y_hi = half["y_range"]
            if x_lo <= point_board[0] <= x_hi and y_lo <= point_board[1] <= y_hi:
                return half_key

        distances = {k: np.linalg.norm(point_board - h["center"])
                     for k, h in self.half_bounds_board.items()}
        return min(distances, key=distances.get)

    def _base_xy_to_board_xy(self, xy_base):
        surface_z_base = transform_point(config.T_BOARD_TO_BASE,
                                          [0.0, 0.0, config.SURFACE_Z_BOARD])[2]
        point_base = np.array([xy_base[0], xy_base[1], surface_z_base])
        point_board = transform_point(self._T_base_to_board, point_base)
        return point_board[:2]

    # ---------- placement decision ----------
    def find_placement(self, half_key, known_chips, color):
        """
        half_key    : 'robot' or 'player' -- the destination half
        known_chips : list of {'position': np.array xyz (base frame),
                      'color': str} for EVERY currently known chip on the
                      board (both halves, both colors) -- used to check
                      for collision-free space and same-color stacking
                      candidates. Chips outside `half_key` are ignored
                      automatically.
        color       : the color of the chip being placed, for the
                      same-color stacking fallback.

        Returns (kind, position_base):
            ('empty', pos) -- pos is flat on the board surface, no
                              existing chip within PLACEMENT_GRID_PITCH
            ('stack', pos) -- pos is directly on top of an existing
                              same-color chip (no empty space was found)
            (None, None)   -- no valid placement in this half right now;
                              caller should try a different chip instead
        """
        half = self.half_bounds_board[half_key]
        pitch = config.PLACEMENT_GRID_PITCH

        # Only chips actually sitting in THIS half matter here. Convert
        # once to board-local xy for cheap repeated distance checks.
        chips_in_half = []
        for c in known_chips:
            xy_board = self._base_xy_to_board_xy(c["position"][:2])
            if (half["x_range"][0] <= xy_board[0] <= half["x_range"][1] and
                    half["y_range"][0] <= xy_board[1] <= half["y_range"][1]):
                chips_in_half.append((xy_board, c["color"], c["position"]))

        # ---------- 1. Collision-free empty space ----------
        x_lo = half["x_range"][0] + config.TOOL_CUP_RADIUS
        x_hi = half["x_range"][1] - config.TOOL_CUP_RADIUS
        y_lo = half["y_range"][0] + config.TOOL_CUP_RADIUS
        y_hi = half["y_range"][1] - config.TOOL_CUP_RADIUS

        for (i, j) in self._spiral_offsets:
            candidate_xy = half["center"] + np.array([i, j]) * pitch
            if not (x_lo <= candidate_xy[0] <= x_hi and y_lo <= candidate_xy[1] <= y_hi):
                continue
            collision = any(np.linalg.norm(candidate_xy - xy_board) < pitch
                             for xy_board, _, _ in chips_in_half)
            if not collision:
                point_board = [candidate_xy[0], candidate_xy[1], config.SURFACE_Z_BOARD]
                return "empty", transform_point(config.T_BOARD_TO_BASE, point_board)

        # ---------- 2. Stack on an existing same-color chip ----------
        for _, chip_color, chip_pos_base in chips_in_half:
            if chip_color == color:
                stacked_pos = np.array(chip_pos_base, dtype=float) + \
                              self.surface_normal_base * config.CHIP_THICKNESS_M
                return "stack", stacked_pos

        # ---------- 3. Nothing available ----------
        return None, None
