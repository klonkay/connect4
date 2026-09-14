"""
task/sorting_task.py
Board-reset state machine. Each color has a FIXED correct half; chips
currently on the wrong half get moved, chips already correct are left
alone. Destination placement is decided by planning/placement_manager.py
based on the board's REAL current occupancy (not an abstract grid):

    1. Collision-free empty space on the destination half -> place there.
    2. No empty space -> stack on an existing same-color chip there.
    3. Neither available -> skip this chip, try a different pending one.

Run cycle: detect all chips -> find the closest misplaced chip that
ALSO has a valid destination available right now -> pick -> place ->
repeat, or return home once nothing is left pending (or nothing pending
has anywhere to go, meaning the board is genuinely full).
"""

import time
import numpy as np

import config
from vision.chip_detector import detect_chips
from planning.path_planner import build_pick_path, build_place_path
from planning.placement_manager import PlacementManager
from planning.geometry_utils import transform_point, transform_vector


class SortingTask:
    def __init__(self, camera, robot, player_color):
        if player_color not in (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR):
            raise ValueError(f"player_color must be '{config.CHIP_ONE_COLOR}' or "
                              f"'{config.CHIP_TWO_COLOR}', got '{player_color}'")
        self.camera = camera
        self.robot = robot
        self.player_color = player_color
        self.robot_color = (config.CHIP_TWO_COLOR if player_color == config.CHIP_ONE_COLOR
                             else config.CHIP_ONE_COLOR)
        self.placement_mgr = PlacementManager()
        self.moved_count = 0
        self.failed_count = 0
        # Real detections have no persistent ID across camera frames, so
        # a chip that exhausts its retries is blacklisted by POSITION
        # instead of name -- any future detection within this tolerance
        # of a blacklisted point is treated as the same physical chip.
        self.unreachable_positions = []

    UNREACHABLE_POSITION_TOL_M = 0.02   # 2cm -- matches a stationary chip between frames

    def _is_blacklisted(self, pos_base):
        return any(np.linalg.norm(pos_base - p) < self.UNREACHABLE_POSITION_TOL_M
                   for p in self.unreachable_positions)

    # ---------- perception ----------
    def _capture_and_detect(self):
        color_image, depth_frame, _ = self.camera.get_frames()
        if color_image is None:
            return []
        return detect_chips(color_image, depth_frame, self.camera)

    def _to_base_frame(self, detection):
        pos_base = transform_point(config.T_CAM_TO_BASE, detection.center_cam)
        normal_base = transform_vector(config.T_CAM_TO_BASE, detection.normal_cam)
        normal_base = normal_base / np.linalg.norm(normal_base)
        return pos_base, normal_base

    def _correct_half_for(self, color_name):
        return "player" if color_name == self.player_color else "robot"

    def _all_known_chips(self, detections):
        """Every currently detected chip (both colors, either half),
        converted to base-frame position -- what placement_manager needs
        to check for collision-free space and stacking candidates."""
        known = []
        for d in detections:
            pos_base, _ = self._to_base_frame(d)
            known.append({"position": pos_base, "color": d.color_name})
        return known

    def _pending_targets(self, detections):
        """Every (detection, pos_base, normal_base, destination_half)
        tuple for a chip currently on the WRONG half for its color.
        Chips already correct, of an unrecognized color, or matching a
        blacklisted unreachable position, are silently skipped."""
        pending = []
        for d in detections:
            if d.color_name not in (self.player_color, self.robot_color):
                continue
            pos_base, normal_base = self._to_base_frame(d)
            if self._is_blacklisted(pos_base):
                continue
            current_half = self.placement_mgr.half_of_point(pos_base[:2])
            correct_half = self._correct_half_for(d.color_name)
            if current_half != correct_half:
                pending.append((d, pos_base, normal_base, correct_half))
        return pending

    def _select_next_valid_target(self, detections):
        """Among all currently misplaced chips (closest-to-camera first),
        returns the first one that also has a valid destination spot
        available right now: (detection, pos_base, normal_base,
        place_position, kind). Returns None if nothing pending has
        anywhere to go (board genuinely full for every pending chip)."""
        pending = self._pending_targets(detections)
        if not pending:
            return None
        known_chips = self._all_known_chips(detections)
        pending_sorted = sorted(pending, key=lambda t: t[0].center_cam[2])
        for detection, pos_base, normal_base, destination_half in pending_sorted:
            kind, place_position = self.placement_mgr.find_placement(
                destination_half, known_chips, detection.color_name)
            if kind is not None:
                return detection, pos_base, normal_base, place_position, kind
        return None

    # ---------- single chip cycle ----------
    def _handle_chip(self, pos_base, normal_base, place_position):
        """Returns True on success, False on any recoverable failure. A
        caught motion exception is treated the same as a failed suction
        attempt -- logged, counted, retried/eventually skipped by run()'s
        loop -- rather than crashing the whole run over one bad chip
        position.

        NOTE: catches a broad Exception rather than a specific ur_rtde
        exception class, since the exact exception types raised by
        rtde_control on an unreachable/faulted move haven't been verified
        against real hardware yet. Once you've seen what ur_rtde actually
        raises in practice, narrow this to those specific exception
        types instead of a bare Exception.
        """
        try:
            pick_path = build_pick_path(pos_base, normal_base)
            self.robot.move_l(pick_path["travel"])
            self.robot.move_l_slow(pick_path["approach"])
            self.robot.suction_on()
            self.robot.move_l_slow(pick_path["contact"])

            if not self.robot.suction_engaged():
                self.robot.suction_off()
                self.robot.move_l(pick_path["retreat"])
                self.failed_count += 1
                return False

            self.robot.move_l(pick_path["retreat"])

            place_path = build_place_path(place_position, self.placement_mgr.surface_normal_base)

            self.robot.move_l(place_path["travel"])
            self.robot.move_l_slow(place_path["approach"])
            self.robot.move_l_slow(place_path["release"])
            self.robot.suction_off()
            self.robot.move_l(place_path["retreat"])

            self.moved_count += 1
            return True

        except Exception as e:
            print(f"    Move failed: {e}")
            self.robot.suction_off()   # in case it was already attached
            self.failed_count += 1
            return False

    # ---------- main loop ----------
    def run(self):
        self.robot.go_home()
        try:
            while True:
                detections = self._capture_and_detect()
                pending = self._pending_targets(detections)

                if not pending:
                    msg = (f"Board reset complete: every reachable '{self.player_color}' chip "
                           f"is on the player's half and every reachable '{self.robot_color}' "
                           f"chip is on the robot's half. Moved={self.moved_count} "
                           f"Failed={self.failed_count}.")
                    if self.unreachable_positions:
                        msg += (f" {len(self.unreachable_positions)} chip(s) could not be "
                                f"reached and were skipped.")
                    print(msg)
                    self.robot.go_home()
                    break

                target = self._select_next_valid_target(detections)
                if target is None:
                    print("No pending chip currently has a valid destination spot -- "
                          "every destination half is full and has no matching-color "
                          "chip to stack on either. Stopping.")
                    self.robot.go_home()
                    break

                detection, pos_base, normal_base, place_position, kind = target
                print(f"Targeting {detection} -> {kind} placement")

                for attempt in range(config.MAX_SORT_RETRIES):
                    if self._handle_chip(pos_base, normal_base, place_position):
                        break
                    print(f"Retry {attempt + 1}/{config.MAX_SORT_RETRIES} for {detection}")
                    detections = self._capture_and_detect()
                    fresh = self._select_next_valid_target(detections)
                    if fresh is None:
                        break
                    detection, pos_base, normal_base, place_position, kind = fresh
                else:
                    print(f"Giving up on {detection} after {config.MAX_SORT_RETRIES} "
                          f"attempts -- marking its position unreachable and skipping "
                          f"it for the rest of this run.")
                    self.unreachable_positions.append(pos_base)

                time.sleep(config.POLL_PERIOD_S)

        except KeyboardInterrupt:
            print("Interrupted by user, retracting and disabling suction.")
        finally:
            self.robot.suction_off()
            self.robot.go_home()
