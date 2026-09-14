"""
simulation/sim_task.py
RoboDK version of the same board-reset logic as task/sorting_task.py,
including real-occupancy-aware placement:

    1. Collision-free empty space on the destination half -> place there.
    2. No empty space -> stack on an existing same-color chip there.
    3. Neither available -> skip this chip, try a different pending one.

Reuses planning/path_planner.py and planning/placement_manager.py
unchanged -- only motion and perception are swapped for the RoboDK
bridge. Chip positions/colors here are ground truth read straight from
the station (via RoboDKInterface.list_chip_items()), standing in for the
real camera + chip_detector.py pipeline.
"""

import time
import numpy as np
from robodk.robolink import StoppedError, TargetReachError

import config
from simulation.robodk_interface import RoboDKInterface
from planning.path_planner import build_pick_path, build_place_path
from planning.placement_manager import PlacementManager


class SimSortingTask:
    def __init__(self, player_color):
        if player_color not in (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR):
            raise ValueError(f"player_color must be '{config.CHIP_ONE_COLOR}' or "
                              f"'{config.CHIP_TWO_COLOR}', got '{player_color}'")
        self.player_color = player_color
        self.robot_color = (config.CHIP_TWO_COLOR if player_color == config.CHIP_ONE_COLOR
                             else config.CHIP_ONE_COLOR)
        self.robodk = RoboDKInterface()
        # PlacementManager reads config.T_BOARD_TO_BASE, which
        # RoboDKInterface.__init__ already overwrote with the ground-truth
        # BoardFrame pose, so construct it after connecting.
        self.placement_mgr = PlacementManager()
        self.moved_count = 0
        self.failed_count = 0
        self.unreachable_names = set()

    def _correct_half_for(self, color):
        return "player" if color == self.player_color else "robot"

    def _pending_targets(self, chips):
        """List of (chip_dict, destination_half) for chips currently on
        the wrong half, excluding unreachable/unrecognized-color ones."""
        pending = []
        for c in chips:
            if c["color"] not in (self.player_color, self.robot_color):
                continue
            if c["item"].Name() in self.unreachable_names:
                continue
            current_half = self.placement_mgr.half_of_point(c["position"][:2])
            correct_half = self._correct_half_for(c["color"])
            if current_half != correct_half:
                pending.append((c, correct_half))
        return pending

    def _select_next_valid_target(self, chips):
        """Among all currently misplaced chips (closest to robot base
        first), returns the first one that also has a valid destination
        spot available right now: (chip_dict, place_position, kind).
        Returns None if nothing pending has anywhere to go."""
        pending = self._pending_targets(chips)
        if not pending:
            return None
        pending_sorted = sorted(pending, key=lambda t: np.linalg.norm(t[0]["position"]))
        for chip, destination_half in pending_sorted:
            kind, place_position = self.placement_mgr.find_placement(
                destination_half, chips, chip["color"])
            if kind is not None:
                return chip, place_position, kind
        return None

    def _handle_chip(self, chip, place_position):
        pos_base = chip["position"]
        normal_base = chip["normal"]

        try:
            pick_path = build_pick_path(pos_base, normal_base)
            self.robodk.move_l(pick_path["travel"])
            self.robodk.move_l_slow(pick_path["approach"])
            self.robodk.move_l_slow(pick_path["contact"])
            self.robodk.suction_on(chip["item"])

            if not self.robodk.suction_engaged():
                self.robodk.suction_off()
                self.robodk.move_l(pick_path["retreat"])
                self.failed_count += 1
                return False

            self.robodk.move_l(pick_path["retreat"])

            place_path = build_place_path(place_position, self.placement_mgr.surface_normal_base)

            self.robodk.move_l(place_path["travel"])
            self.robodk.move_l_slow(place_path["approach"])
            self.robodk.move_l_slow(place_path["release"])
            self.robodk.suction_off()
            self.robodk.move_l(place_path["retreat"])

            self.moved_count += 1
            return True

        except (StoppedError, TargetReachError) as e:
            print(f"    Move failed for {chip['item'].Name()}: {e}")
            self.robodk.suction_off()
            self.failed_count += 1
            return False

    def run(self):
        self.robodk.go_home()
        while True:
            chips = self.robodk.list_chip_items()
            pending = self._pending_targets(chips)

            if not pending:
                msg = (f"Board reset complete: every reachable '{self.player_color}' chip is "
                       f"on the player's half and every reachable '{self.robot_color}' chip is "
                       f"on the robot's half. Moved={self.moved_count} Failed={self.failed_count}.")
                if self.unreachable_names:
                    msg += (f" {len(self.unreachable_names)} chip(s) could not be reached and "
                            f"were skipped: {sorted(self.unreachable_names)}.")
                print(msg)
                self.robodk.go_home()
                break

            target = self._select_next_valid_target(chips)
            if target is None:
                print("No pending chip currently has a valid destination spot -- "
                      "every destination half is full and has no matching-color "
                      "chip to stack on either. Stopping.")
                self.robodk.go_home()
                break

            chip, place_position, kind = target
            print(f"Targeting {chip['color']} chip: {chip['item'].Name()} -> {kind} placement")

            for attempt in range(config.MAX_SORT_RETRIES):
                if self._handle_chip(chip, place_position):
                    break
                print(f"Retry {attempt + 1}/{config.MAX_SORT_RETRIES} for {chip['item'].Name()}")
                chips = self.robodk.list_chip_items()
                fresh = self._select_next_valid_target(chips)
                if fresh is None:
                    break
                chip, place_position, kind = fresh
            else:
                print(f"Giving up on {chip['item'].Name()} after {config.MAX_SORT_RETRIES} "
                      f"attempts -- marking unreachable and skipping it for the rest of this run.")
                self.unreachable_names.add(chip["item"].Name())

            time.sleep(config.POLL_PERIOD_S)


def _prompt_for_color():
    options = (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR)
    while True:
        choice = input(f"Which color is the player using? {options}: ").strip().lower()
        if choice in options:
            return choice
        print(f"'{choice}' isn't one of {options}, try again.")


if __name__ == "__main__":
    color = _prompt_for_color()
    SimSortingTask(color).run()
