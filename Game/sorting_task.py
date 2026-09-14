"""
sorting_task.py
----------------
Sorts chips by colour into two halves of the board.
Supports stacking if a half is full, and can reorder sorting of the other colour.
"""

import time
import numpy as np

import config
import sorting_config
from interfaces import ArmMotionError

class SortingTask:
    def __init__(self, camera, arm, player_color, verbose=True):
        self.camera = camera
        self.arm = arm
        self.player_color = player_color.lower()
        self.robot_color = 'yellow' if self.player_color == 'red' else 'red'
        self.verbose = verbose

        self.T_board_to_base = config.T_BOARD_TO_BASE
        self.T_base_to_board = np.linalg.inv(self.T_board_to_base)
        self.half_rects = self._compute_half_rects()
        self.occupancy = {'player': {}, 'robot': {}}

    def _compute_half_rects(self):
        """
        Rectangles in BOARD-LOCAL coordinates, taken directly from
        sorting_config.py -- deliberately NOT transformed to base frame
        here. Keeping them in board-local space means the two halves'
        boundaries can never be distorted by T_board_to_base's rotation,
        no matter how large that rotation is.

        The previous approach transformed these rectangles INTO base
        frame and checked chip positions against the transformed result
        -- but a rotated rectangle's true footprint isn't itself an
        axis-aligned rectangle, so any such transform-then-bound approach
        distorts the region. Testing this confirmed the two halves
        started overlapping at just ~2 degrees of board rotation --
        something close to guaranteed on any real, ArUco-calibrated
        mount. Checking membership by inverse-transforming the CHIP'S
        POSITION back into board-local space instead (see
        _is_inside_half) is mathematically equivalent for a
        non-rotated board, but immune to this problem for a rotated one.
        """
        rects = {}
        for half, corners in [('player', sorting_config.SORT_PLAYER_HALF_CORNERS),
                              ('robot', sorting_config.SORT_ROBOT_HALF_CORNERS)]:
            (x1, y1, z1), (x2, y2, z2) = corners
            rects[half] = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        return rects

    def _base_to_board_xy(self, x, y, z=0.0):
        """Inverse-transforms a BASE FRAME point into BOARD-LOCAL (x, y)."""
        p = np.array([x, y, z, 1.0])
        board_pt = self.T_base_to_board @ p
        return board_pt[0], board_pt[1]

    def _board_to_base_xy(self, bx, by, bz=0.0):
        """Forward-transforms a BOARD-LOCAL point into BASE FRAME (x, y)."""
        p = np.array([bx, by, bz, 1.0])
        base_pt = self.T_board_to_base @ p
        return base_pt[0], base_pt[1]

    def _is_inside_half(self, x, y, half, z=0.0):
        """
        x, y, z are in BASE FRAME (matching how detected chip positions
        are used everywhere else in this file). Inverse-transforms into
        board-local coordinates before checking against the
        never-rotated rectangle -- see _compute_half_rects()'s docstring
        for why this direction (rather than transforming the rectangle)
        is what makes this immune to rotation-induced overlap.
        """
        bx, by = self._base_to_board_xy(x, y, z)
        xmin, xmax, ymin, ymax = self.half_rects[half]
        return (xmin <= bx <= xmax) and (ymin <= by <= ymax)

    def _is_correctly_placed(self, x, y, color):
        """
        Returns True if a chip of this color, at its CURRENT position, is
        already within its correct half -- meaning it doesn't need to be
        picked up and moved at all. Checked in run() BEFORE any
        pick-and-place is attempted, so correctly-placed chips are left
        alone instead of being needlessly re-sorted into a new slot
        within the same half.
        """
        target_half = 'player' if color == self.player_color else 'robot'
        return self._is_inside_half(x, y, target_half)

    def _find_placement(self, half):
        xmin, xmax, ymin, ymax = self.half_rects[half]
        pitch = sorting_config.PLACEMENT_GRID_PITCH
        radius = sorting_config.CHIP_RADIUS
        # Candidates generated directly in BOARD-LOCAL space -- always a
        # clean, undistorted axis-aligned grid regardless of
        # T_board_to_base's rotation. Each is forward-transformed to
        # base frame below, only once a spot is actually chosen.
        xs = np.arange(xmin + radius, xmax - radius, pitch)
        ys = np.arange(ymin + radius, ymax - radius, pitch)
        candidates_board = [(bx, by) for bx in xs for by in ys]

        # Occupancy keys are stored in BASE FRAME (matching detected chip
        # positions -- see run()'s "already correctly placed" registration
        # and this method's own return value below), so convert each
        # board-local candidate to base frame before the avoidance check.
        candidates = []
        for bx, by in candidates_board:
            base_x, base_y = self._board_to_base_xy(bx, by)
            candidates.append((bx, by, base_x, base_y))

        occ = self.occupancy[half]
        candidates = [c for c in candidates
                      if all(np.linalg.norm(np.array(c[2:4]) - np.array([ox, oy])) > 2 * radius
                             for (ox, oy) in occ)]

        if not candidates:
            raise RuntimeError(f"No free spot in {half} half.")

        # Prefer the candidate closest to the center of the half,
        # measured in board-local space (consistent with how the
        # rectangle itself is defined).
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        candidates.sort(key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        best_bx, best_by, best_x, best_y = candidates[0]

        p0 = np.array([sorting_config.SORT_PLAYER_HALF_CORNERS[0][0],
                       sorting_config.SORT_PLAYER_HALF_CORNERS[0][1],
                       0.02, 1.0])
        p0_base = self.T_board_to_base @ p0
        surface_z = p0_base[2]

        stack_height = len(occ)
        z = surface_z + stack_height * sorting_config.STACK_OFFSET_Z

        self.occupancy[half][(best_x, best_y)] = stack_height + 1
        return best_x, best_y, z

    def _sort_chip(self, chip_pose, chip_color):
        target_half = 'player' if chip_color == self.player_color else 'robot'
        other_half = 'robot' if target_half == 'player' else 'player'

        try:
            place_x, place_y, place_z = self._find_placement(target_half)
        except RuntimeError:
            occ = self.occupancy[target_half]
            if not occ:
                raise
            best_spot = min(occ.items(), key=lambda kv: kv[1])
            (ox, oy), height = best_spot
            new_height = height + 1
            self.occupancy[target_half][(ox, oy)] = new_height
            p0 = np.array([sorting_config.SORT_PLAYER_HALF_CORNERS[0][0],
                           sorting_config.SORT_PLAYER_HALF_CORNERS[0][1],
                           0.0, 1.0])
            p0_base = self.T_board_to_base @ p0
            surface_z = p0_base[2]
            z = surface_z + (new_height - 1) * sorting_config.STACK_OFFSET_Z
            place_x, place_y, place_z = ox, oy, z

        place_pose = [place_x, place_y, place_z, 2.905, 1.199, 0]    # flat orientation
        try:
            self.arm.pick_and_place_to_pose(chip_pose, place_pose)
            return True
        except ArmMotionError as e:
            self._log(f"Pick-and-place failed: {e}")
            return False

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self):
        """
        Iteratively re-captures chip positions after EVERY pick-and-place
        attempt, rather than working off a single snapshot taken once at
        the start of the whole session. This is the actual fix for two
        related problems with the old approach:
          - if a pick-and-place attempt genuinely failed (missed grip,
            suction slip, dropped in the wrong spot), the old code kept
            retrying the SAME stale pickup pose blindly -- it never
            re-checked where the chip actually ended up. Now, every pass
            re-captures the real board state, so a failed attempt is
            re-approached using the chip's CURRENT actual position, not
            a stale one from before the attempt.
          - a placement that DID succeed is correctly recognized as such
            on the next capture (via _is_correctly_placed) and left
            alone, rather than being touched again.

        No global pass cap: a genuine worst case can require as many
        passes as there are chips on the board (up to 42), so a fixed
        max_passes would be wrong either way -- too low for a legitimate
        full-board sort, or not actually protective against a stuck chip
        if set high enough to allow one.

        Instead, each chip's pick-and-place attempts are tracked
        INDIVIDUALLY, keyed by its approximate position (detected chip
        IDs aren't guaranteed stable across separate camera captures, so
        rounded position is used as a practical stand-in for "the same
        physical chip" across passes -- reasonable since a chip that
        fails to move should still be roughly where it was).
        sorting_config.MAX_RETRIES_PER_CHIP caps TOTAL attempts on that
        one chip across ALL passes (not a burst of retries within a
        single pass, each still using a fresh capture) -- once a
        specific chip hits that cap, it's skipped in favor of trying
        OTHER misplaced chips, so one stuck/unreachable chip can't block
        progress on chips that ARE sortable. The loop only ends once
        either nothing is left to sort, or every remaining misplaced
        chip has exhausted its own attempt budget.

        Requires the arm to actually be clear of the camera's view
        (home) by the time each capture happens -- pick_and_place_to_pose
        should already leave it there; SORT_SETTLE_TIME below adds a
        short pause before each capture to be safe, matching the same
        "let the arm clear the view before trusting a frame" pattern
        used elsewhere in this project (game_manager.py's
        robot_settle_time).
        """
        settle_time = getattr(sorting_config, 'SORT_SETTLE_TIME', 1.5)
        max_attempts_per_chip = sorting_config.MAX_RETRIES_PER_CHIP

        self._log(f"Sorting: player={self.player_color}, robot={self.robot_color}")

        # Total attempts made so far on a given chip, keyed by its
        # rounded (x, y) position.
        attempt_counts = {}

        def position_key(x, y):
            return (round(x, 3), round(y, 3))

        pass_num = 0
        while True:
            pass_num += 1
            self._log(f"\n--- Pass {pass_num}: capturing chip positions ---")
            time.sleep(settle_time)

            # ---- Grab a FRESH frame and detect chips -- this happens at
            # the top of EVERY pass, not once at the start of run().
            if hasattr(self.camera, 'update'):
                self.camera.update()
            raw_chips = self.camera.get_chip_positions()

            if not raw_chips:
                self._log("No chips detected.")
                if pass_num == 1:
                    return
                break   # nothing left to see -- treat remaining chips as done

            chips = []
            for chip in raw_chips:
                x, y, z = chip.position_m
                chips.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'color': chip.color,
                    'pixel': chip.pixel,
                    'id': chip.id
                })

            # ---- Find chips that are NOT yet correctly placed, using
            # THIS pass's fresh positions.
            misplaced = [c for c in chips if c.get('color') and
                         not self._is_correctly_placed(c['x'], c['y'], c['color'])]

            if not misplaced:
                self._log("All detected chips are correctly placed. Sorting complete.")
                return

            # ---- Skip chips that have already exhausted their attempt
            # budget, so a stuck one can't block progress on the rest.
            sortable_now = [c for c in misplaced
                            if attempt_counts.get(position_key(c['x'], c['y']), 0)
                            < max_attempts_per_chip]

            if not sortable_now:
                stuck_positions = [position_key(c['x'], c['y']) for c in misplaced]
                self._log(f"{len(misplaced)} chip(s) remain misplaced but have each "
                          f"failed {max_attempts_per_chip} times -- stopping. "
                          f"Stuck positions: {stuck_positions}")
                return

            self._log(f"{len(misplaced)} misplaced ({len(sortable_now)} still "
                      f"within their attempt budget).")

            # ---- Attempt exactly ONE chip this pass -- the next pass
            # re-captures and re-checks everything fresh regardless of
            # whether this attempt succeeds.
            chip = sortable_now[0]
            color = chip['color']
            key = position_key(chip['x'], chip['y'])
            attempt_num = attempt_counts.get(key, 0) + 1
            self._log(f"Processing {color} chip (id={chip['id']}), "
                      f"attempt {attempt_num}/{max_attempts_per_chip}.")
            pickup_pose = [chip['x'], chip['y'], chip['z'], 2.905, 1.199, 0]

            success = False
            try:
                success = self._sort_chip(pickup_pose, color)
            except RuntimeError:
                other_color = self.robot_color if color == self.player_color else self.player_color
                self._log(f"Target half full. Trying to sort a {other_color} chip first.")
                other_chip = next((c for c in chips if c.get('color') == other_color), None)
                if other_chip is None:
                    self._log(f"No {other_color} chip left to free space.")
                else:
                    other_pose = [other_chip['x'], other_chip['y'], other_chip['z'], 3.14, 0.0, 0.0]
                    try:
                        other_success = self._sort_chip(other_pose, other_color)
                        if other_success:
                            self._log(f"Sorted {other_color} chip, freeing space.")
                    except Exception as ex:
                        self._log(f"Failed to sort other chip: {ex}")

            if not success:
                attempt_counts[key] = attempt_num
                self._log(f"Failed to place chip this pass "
                          f"({attempt_num}/{max_attempts_per_chip} attempts used) -- "
                          f"will re-check its actual position next pass rather than "
                          f"retrying blindly.")