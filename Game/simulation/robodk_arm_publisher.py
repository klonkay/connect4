"""
simulation/robodk_arm_publisher.py
A drop-in replacement for ur3_interface.UR3SuctionArmPublisher, backed by
a RoboDK simulation instead of real urx/hardware. Mirrors its public
method names, signatures, and move sequence exactly, so game_manager.py,
sorting_task.py, and main_windows.py can all use this unmodified in place
of the real class.

Key differences from the real class, both deliberate:
  - suction_on()/suction_off() don't toggle a digital output; they
    attach/detach the nearest chip object to the tool's frame in RoboDK
    (see robodk_interface.py's suction_on/off for exactly how).
  - suction_on() is still zero-argument for interface parity, but
    internally finds whichever chip is currently closest to the tool's
    TCP -- the simulation equivalent of "whatever the cup happens to be
    touching when the valve opens."
"""

import time
import numpy as np
from robodk.robolink import StoppedError, TargetReachError

from interfaces import ArmPublisher, ArmMotionError
from board import PLAYER, ROBOT
import sorting_config
from config import COLUMN_POSITIONS, GAME_ORIENTATION, HOME_JOINTS

from simulation.robodk_interface import RoboDKInterface


class RoboDKSuctionArmPublisher(ArmPublisher):
    # How close a chip needs to be to the tool's current TCP to count as
    # "the one being picked up" when suction_on() is called.
    PICK_PROXIMITY_TOL_M = 0.01

    def __init__(self,
                 home_position: list = HOME_JOINTS,
                 approach_height: float = 0.2,
                 pickup_lower_offset: float = 0.001,
                 pickup_positions: dict = None,
                 column_position_provider=None):
        """
        Args (mirroring UR3SuctionArmPublisher where the concept still
        applies -- robot_ip/suction_pin/speed/acceleration are accepted
        but unused, kept only so existing construction code can pass them
        without erroring):
            home_position: joint angles for home (radians) -- passed
                straight to config.HOME_JOINTS by default, RoboDK just
                uses whatever robodk_interface.py's go_home() resolves.
            approach_height: extra Z clearance used specifically for
                column-drop hovering, matching the real class's
                'approach_height' constructor param (separate from
                sorting_config.SORT_APPROACH_HEIGHT, which governs the
                pick/place approach legs inside pick_and_place_to_pose).
            pickup_lower_offset: how far below the pickup pose's nominal
                Z the tool lowers to actually grab the chip (mirrors the
                real class's hardcoded -0.003 in pick_and_place_to_pose).
            pickup_positions: fallback fixed positions for chips (dict
                {chip_const: [x,y,z]}) -- kept for interface parity;
                unlikely to matter here since chip positions always come
                from ground truth, not a "couldn't find it" fallback.
            column_position_provider: optional callable(col) -> (x, y, z)
                in robot base frame, used INSTEAD of the normal
                COLUMN_POSITIONS + T_BOARD_TO_BASE transform. Mainly
                useful in simulation as a cross-check: set this to
                camera.get_column_entrance_position after constructing
                your RoboDKCellCamera, and compare its result against
                the transform-based one -- if they disagree by more than
                a few mm, your BoardFrame item's pose likely doesn't
                match the real board's geometry yet. Leave as None for
                normal operation (matches the real UR3SuctionArmPublisher's
                behavior, just with a corrected, working transform).
        """
        self.home_position = home_position
        self.approach_height = approach_height
        self.pickup_lower_offset = pickup_lower_offset
        self.column_position_provider = column_position_provider

        # Matches ur3_interface.py's fixed_rxryrz exactly -- the
        # orientation used for pickup and for 'sort'-orientation column
        # poses (straight-down suction).
        self.fixed_rxryrz = [2.905, 1.199, 0.0]

        self.pickup_positions = pickup_positions or {
            PLAYER: [0.30, 0.20, 0.10],
            ROBOT:  [0.30, -0.20, 0.10],
        }

        # Set right before suction_off() in pick_and_place_to_pose(), so
        # callers (e.g. the harness) know EXACTLY which chip item was
        # just released, instead of guessing by proximity after the fact.
        self.last_placed_chip_item = None

        self.rdk_iface = RoboDKInterface()
        self.rdk_iface.suction_off()
        self._go_home()

    # ---------- motion commands ----------
    def _go_home(self):
        print("Moving to home...")
        self.rdk_iface.go_home()

    def _move_to_pose(self, pose, is_joint=False, slow=False):
        """Move to a Cartesian pose [x,y,z,rx,ry,rz], meters/radians.
        is_joint is accepted for interface parity but this project's
        RoboDK bridge only exposes Cartesian moves for this flow (the
        real class also only ever calls this with is_joint=False from
        pick_and_place_to_pose). slow=True uses
        config.APPROACH_LIN_SPEED/ACCEL instead of the normal travel
        speed -- intended for the final descent into contact, where
        precision matters more than speed."""
        try:
            self.rdk_iface.move_l(pose, slow=slow)
        except (StoppedError, TargetReachError) as e:
            try:
                self._go_home()
            except Exception:
                pass
            raise ArmMotionError(f"Motion to pose {pose} failed: {e}")

    # ---------- board-relative column poses ----------
    def _get_column_drop_pose(self, col, approach=False, orientation='sort'):
        """Builds the drop pose for a column.

        config.COLUMN_POSITIONS are BOARD-LOCAL coordinates (measured
        directly from the physical board's geometry) -- config.py's
        comment calling them "robot base frame" is simply wrong/stale
        and should be corrected there. Transforming them through
        config.T_BOARD_TO_BASE is the correct, and only, approach that
        works in both real and simulated contexts without maintaining
        two separate hardcoded coordinate sets.

        This REQUIRES T_BOARD_TO_BASE to actually reflect where the
        board currently is. On real hardware that comes from your ArUco
        board calibration. In simulation, robodk_interface.py already
        overwrites config.T_BOARD_TO_BASE from a 'BoardFrame' item's
        ground-truth pose if one exists in the station -- if it doesn't
        exist, T_BOARD_TO_BASE silently stays whatever was last
        calibrated for a DIFFERENT (real) setup, and every column
        position computed here will be correspondingly wrong. Make sure
        your simulation station has a BoardFrame item positioned to
        match the same coordinate convention COLUMN_POSITIONS was
        measured in.

        If column_position_provider is set (see __init__), it's used
        INSTEAD of this transform -- useful mainly as a cross-check in
        simulation: if it disagrees with the transform-based result
        below by more than a few mm, that's a strong signal your
        BoardFrame item's position/orientation doesn't actually match
        reality yet.
        """
        if self.column_position_provider is not None:
            x, y, z = self.column_position_provider(col)
        else:
            import config as _config   # re-import to pick up any T_BOARD_TO_BASE
                                         # override RoboDKInterface applied from BoardFrame
            x_b, y_b, z_b = COLUMN_POSITIONS[col]
            point_board = np.array([x_b, y_b, z_b, 1.0])
            point_base = _config.T_BOARD_TO_BASE @ point_board
            x, y, z = point_base[0], point_base[1], point_base[2]

        if approach:
            z += self.approach_height

        if orientation == 'game':
            rx, ry, rz = GAME_ORIENTATION
        else:
            rx, ry, rz = self.fixed_rxryrz

        return [x, y, z, rx, ry, rz]

    # ---------- suction ----------
    def suction_on(self):
        """Zero-argument for interface parity with the real class.
        Finds whichever chip is currently closest to the tool's TCP and
        attaches it -- see robodk_interface.py's suction_on() for the
        actual reparenting mechanics."""
        print("[Suction] ON (sim)")
        tcp_pos = np.array(self.rdk_iface.current_pose()[:3])
        chips = self.rdk_iface.list_chip_items()
        if not chips:
            raise ArmMotionError("No chips found in the station to pick up.")
        nearest = min(chips, key=lambda c: np.linalg.norm(c["position"] - tcp_pos))
        dist = np.linalg.norm(nearest["position"] - tcp_pos)
        if dist > self.PICK_PROXIMITY_TOL_M:
            raise ArmMotionError(
                f"No chip close enough to the tool to pick up "
                f"(nearest is {dist * 1000:.1f}mm away, need < "
                f"{self.PICK_PROXIMITY_TOL_M * 1000:.0f}mm).")
        self.rdk_iface.suction_on(nearest["item"])

    def suction_off(self):
        print("[Suction] OFF (sim)")
        self.rdk_iface.suction_off()

    # ---------- generic pick-and-place (mirrors the real class exactly) ----------
    def pick_and_place_to_pose(self, pickup_pose, place_pose, wrist_yaw=None):
        """
        Pick from pickup_pose and place at place_pose. Both poses are
        [x,y,z,rx,ry,rz] in robot base frame.

        wrist_yaw: optional override for the Z-rotation component used
        DURING THE APPROACH LEGS ONLY (e.g. to avoid cable wrap). Leave
        as None (default) to keep the SAME orientation throughout the
        whole approach-then-descend sequence, matching pickup_pose/
        place_pose's own rotation exactly -- this is almost always what
        you want, since it means the arm arrives at the hover point
        already correctly oriented and only translates straight down
        from there, rather than hovering at some other orientation and
        rotating into place during the final descent.

        NOTE: the original real-hardware implementation this was copied
        from always overwrote the approach pose's rz with wrist_yaw
        (defaulting to 0.0), which silently discarded the actual
        intended orientation (e.g. GAME_ORIENTATION's rz) on every
        single move, regardless of whether a wrist_yaw override was ever
        actually wanted. Fixed here so it only applies when explicitly
        requested.
        """
        try:
            # 1. Approach pickup -- simple world-Z offset, same as before.
            #    This was already correct; only the place/column side
            #    needs orientation-aware retreat (see step 5), since
            #    GAME_ORIENTATION specifically is a steep, non-vertical
            #    rotation. Applying that same logic to pickup's own
            #    orientation ([2.905, 1.199, 0.0]) computed a different,
            #    apparently insufficient retreat and broke pickup
            #    clearance, so it's kept separate here deliberately.
            approach = list(pickup_pose)
            approach[2] = approach[2] + sorting_config.SORT_APPROACH_HEIGHT
            if wrist_yaw is not None:
                approach[5] = wrist_yaw
            self._move_to_pose(approach)

            # 2. Lower to pickup -- slow, linear descent for precision
            pick = list(pickup_pose)
            pick[2] = pick[2] - self.pickup_lower_offset
            self._move_to_pose(pick, slow=True)

            # 3. Suction ON
            self.suction_on()

            # 4. Lift
            self._move_to_pose(approach)

            # 5. Move to place approach -- simple world-Z offset, kept
            #    purely vertical. A Connect Four column is a vertical
            #    slot in WORLD space, entered from directly above,
            #    regardless of whatever angle GAME_ORIENTATION happens to
            #    rotate the tool to -- that orientation likely exists for
            #    an unrelated reason (matching the tool's physical mount
            #    angle, avoiding a wrist singularity), not to define the
            #    direction of travel into the column. Retreating along
            #    the tool's own axis here was wrong: for a steep rotation
            #    like GAME_ORIENTATION, it moved mostly SIDEWAYS instead
            #    of upward, causing a horizontal approach into the board
            #    instead of a vertical drop into the slot.
            place_approach = list(place_pose)
            place_approach[2] = place_approach[2] + sorting_config.SORT_APPROACH_HEIGHT
            if wrist_yaw is not None:
                place_approach[5] = wrist_yaw
            self._move_to_pose(place_approach)

            # 6. Lower to place -- slow, linear descent for precision
            # (this is the column-entrance-to-final-drop-position leg)
            self._move_to_pose(list(place_pose), slow=True)

            # 7. Suction OFF
            self.last_placed_chip_item = self.rdk_iface._held_item
            self.suction_off()
            print("Chip released.")

            # 8. Retract
            self._move_to_pose(place_approach)
            self._go_home()

        except Exception as e:
            try:
                self.suction_off()
                self._go_home()
            except Exception:
                pass
            raise ArmMotionError(f"Pick-and-place failed: {e}")

    # ---------- game-specific move (uses fixed pickup positions) ----------
    def publish_move(self, row: int, col: int, chip: int):
        """Required by the ArmPublisher ABC. game_manager.py doesn't
        actually call this in the current flow (it calls
        pick_and_place_to_pose + _get_column_drop_pose directly), but
        it's implemented for interface completeness, mirroring the real
        class's fallback-pickup-position behavior."""
        pickup_pose_xyz = self.pickup_positions.get(chip)
        if pickup_pose_xyz is None:
            raise ArmMotionError(f"No pickup position for chip {chip}")
        pickup_pose = pickup_pose_xyz + self.fixed_rxryrz
        drop_pose = self._get_column_drop_pose(col, approach=False, orientation='game')
        self.pick_and_place_to_pose(pickup_pose, drop_pose)

    # ---------- cleanup ----------
    def shutdown(self):
        try:
            self.suction_off()
        except Exception:
            pass
