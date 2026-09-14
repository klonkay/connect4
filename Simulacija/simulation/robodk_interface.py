"""
simulation/robodk_interface.py
Bridges the real sorting algorithm to a RoboDK station for simulation,
without touching any physical hardware or a real camera. This module
takes the place of robot/robot_controller.py *and* vision/camera_interface.py
+ vision/chip_detector.py combined: motion goes through the RoboDK API
instead of ur_rtde, and "detection" reads ground-truth poses/colors
directly from the station instead of a real RGB-D camera. This lets you
validate path planning, the tool-dimension math, and the sorting state
machine before ever touching the physical robot.

IMPORTANT UNIT QUIRK: robodk.robomath.UR_2_Pose() / Pose_2_UR() treat the
translation part as MILLIMETERS (matching RoboDK's internal convention)
even though real UR controllers report/accept poses in METERS. This is a
known mismatch, not a typo -- see the RoboDK forum. Since path_planner.py
outputs meters (to match the real robot code), this module explicitly
converts m -> mm before calling UR_2_Pose(). Don't remove that scaling.

Expected station setup:
  - A robot item named to match ROBODK_ROBOT_NAME (e.g. the UR3 dragged
    in from the RoboDK library).
  - A tool item named to match ROBODK_TOOL_NAME, attached to the robot's
    flange. Its own TCP setting inside RoboDK does NOT need to be
    manually configured -- config.TOOL_TCP_OFFSET is pushed in as the
    robot's active TCP automatically below, the same way
    robot/robot_controller.py pushes it into the real UR controller via
    rtde_c.setTcp(). config.py is the single source of truth either way.
  - Optionally, a Target item named ROBODK_HOME_TARGET for the resting
    position; otherwise config.HOME_JOINTS is used directly.
  - One station object per chip, named "chip_<color>_<n>", e.g.
    "chip_red_1", "chip_blue_1" -- scattered across your surface model
    at realistic (possibly tilted) poses. The chip model's local +Z axis
    should point out of its flat face; that axis is used as the ground-
    truth "surface normal" in place of a real camera's depth-fit normal.
"""

import numpy as np
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_OBJECT
from robodk import robomath

import config

ROBODK_ROBOT_NAME = "UR3"
ROBODK_TOOL_NAME = "SuctionTool"
ROBODK_HOME_TARGET = "Home"   # Target item name; falls back to config.HOME_JOINTS if absent
BOARD_FRAME_NAME = "BoardFrame"   # optional Frame item marking the destination surface's origin


class RoboDKInterface:
    def __init__(self):
        self.rdk = Robolink()

        self.robot = self.rdk.Item(ROBODK_ROBOT_NAME, ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise RuntimeError(
                f"Robot '{ROBODK_ROBOT_NAME}' not found. Open your RoboDK station "
                f"first, or update ROBODK_ROBOT_NAME to match your robot item's name.")

        self.tool = self.rdk.Item(ROBODK_TOOL_NAME, ITEM_TYPE_TOOL)
        if not self.tool.Valid():
            print(f"Warning: tool '{ROBODK_TOOL_NAME}' not found. Motion will still "
                  f"work but suction attach/detach will parent chips directly to the "
                  f"robot flange instead of a tool item.")

        # Push config.TOOL_TCP_OFFSET in as the robot's active TCP,
        # regardless of whatever TCP the tool item itself might have set
        # in RoboDK's own dialog -- config.py is the single source of
        # truth. This mirrors robot_controller.py's set_tool_tcp() on the
        # real robot exactly, so simulated and real paths always agree
        # once you've measured the real tool.
        self.robot.setPoseTool(self._ur6_to_robodk_pose(config.TOOL_TCP_OFFSET))

        # Robot items in RoboDK: Pose()/PoseAbs() on the ROBOT ITSELF
        # return the CURRENT forward-kinematics result (i.e. wherever the
        # TCP happens to be right now) -- not the robot's fixed mounting
        # location. Using that as "the base frame" would make every
        # chip-position calculation drift with the arm's current pose,
        # which is exactly what was happening. The robot's actual static
        # base is its PARENT item's absolute pose, which never changes
        # during a run, so it's computed once and cached here.
        self._robot_base_pose_abs = self.robot.Parent().PoseAbs()

        # Critical: a robot's ACTIVE REFERENCE FRAME (what MoveL/MoveJ
        # poses are interpreted relative to) is a separate setting from
        # the station-tree parent used above -- RoboDK does not assume
        # they're the same thing. Without this line, our math and the
        # actual applied motion could each be internally consistent yet
        # still disagree with each other, since they'd silently be using
        # two different reference frames. Linking it directly to the
        # parent item (rather than a snapshot pose) keeps them locked
        # together even if that frame item is later moved.
        self.robot.setPoseFrame(self.robot.Parent())

        self.robot.setSpeed(config.DEFAULT_LIN_SPEED * 1000)      # RoboDK API takes mm/s
        self.robot.setAcceleration(config.DEFAULT_LIN_ACCEL * 1000)

        self._held_item = None

        # Turns on RoboDK's own collision engine for the whole station.
        # Combined with checking self.rdk.Collisions() after every move
        # (see move_l() below), this catches the arm hitting your
        # imported board assembly (or anything else in the Collision
        # Map) even though our own path planning only reasons about a
        # single known obstruction height, not full 3D geometry. Make
        # sure your board object is included in RoboDK's Collision Map
        # (Tools > Collision Map) -- newly added objects usually are by
        # default, but it's worth checking.
        self.rdk.setCollisionActive(1)

        # Overwrite config.T_BOARD_TO_BASE with the ground-truth pose of a
        # 'BoardFrame' item in the station, if one exists. This replaces
        # board_calibration.py's ArUco detection for simulation purposes --
        # RoboDK already knows exactly where everything is, so there's no
        # need to fake a marker. If no such frame exists, T_BOARD_TO_BASE
        # is left as whatever config.py already has (identity by default),
        # meaning BOARD_ROBOT_HALF_CORNER_A/B and BOARD_PLAYER_HALF_CORNER_A/B
        # are then interpreted directly in the robot's base frame -- fine
        # for a quick test with no explicit board object.
        config.T_BOARD_TO_BASE = self.board_to_base_transform()

    @staticmethod
    def _mat_to_transform_m(mat):
        """Converts a RoboDK Mat (position in mm) to a 4x4 numpy transform
        in meters, matching the convention used everywhere else in the
        project (config.T_CAM_TO_BASE, config.T_BOARD_TO_BASE, etc)."""
        T = np.eye(4)
        T[:3, 0] = mat.VX()
        T[:3, 1] = mat.VY()
        T[:3, 2] = mat.VZ()
        T[:3, 3] = np.array(mat.Pos()) / 1000.0
        return T

    def board_to_base_transform(self):
        """Reads a Frame item named BOARD_FRAME_NAME, if present, and
        returns its pose relative to the robot base as a 4x4 numpy
        transform in meters -- this is what config.BOARD_ROBOT_HALF_CORNER_A/B
        and BOARD_PLAYER_HALF_CORNER_A/B get measured relative to. Place
        this Frame item anywhere on your simulated surface (a corner is
        usually easiest to measure from) with its axes aligned however
        you want the surface's local X/Y to run."""
        frame = self.rdk.Item(BOARD_FRAME_NAME)
        if not frame.Valid():
            print(f"No '{BOARD_FRAME_NAME}' item found in the station -- "
                  f"the board-local corner values in config.py will be "
                  f"interpreted directly in the robot base frame.")
            return np.eye(4)
        pose_in_base = self._robot_base_pose_abs.inv() * frame.PoseAbs()
        return self._mat_to_transform_m(pose_in_base)

    # ---------- motion ----------
    def go_home(self):
        target = self.rdk.Item(ROBODK_HOME_TARGET)
        if target.Valid():
            self.robot.MoveJ(target)
        else:
            self.robot.MoveJ([np.degrees(j) for j in config.HOME_JOINTS])

    def _ur6_to_robodk_pose(self, pose_ur6):
        """pose_ur6: [x, y, z, rx, ry, rz] in METERS/radians -- the exact
        format planning/path_planner.py already produces. Converts to a
        RoboDK Mat via UR_2_Pose(), scaling m -> mm first (see module
        docstring for why that scaling is required)."""
        x, y, z, rx, ry, rz = pose_ur6
        return robomath.UR_2_Pose([x * 1000, y * 1000, z * 1000, rx, ry, rz])

    def move_l(self, pose_ur6, slow=False):
        mat = self._ur6_to_robodk_pose(pose_ur6)
        speed = config.APPROACH_LIN_SPEED if slow else config.DEFAULT_LIN_SPEED
        self.robot.setSpeed(speed * 1000)
        self.robot.MoveL(mat)
        n_collisions = self.rdk.Collisions()
        if n_collisions > 0:
            print(f"WARNING: {n_collisions} collision pair(s) detected after this move -- "
                  f"check the 3D view. Likely causes: BOARD_OBSTRUCTION_HEIGHT set too low "
                  f"for your real board geometry, or the board object isn't where "
                  f"config.T_BOARD_TO_BASE thinks it is.")

    def move_l_slow(self, pose_ur6):
        self.move_l(pose_ur6, slow=True)

    @staticmethod
    def _pose_close(pose_a, pose_b, tol_mm=0.5):
        pa, pb = pose_a.Pos(), pose_b.Pos()
        return sum((a - b) ** 2 for a, b in zip(pa, pb)) ** 0.5 < tol_mm

    def _reparent_preserving_world_pose(self, item, new_parent):
        """Reparents `item` onto `new_parent`, preserving its exact world
        pose. The standard approach -- compute the needed local pose as
        new_parent.PoseAbs().inv() * target_world_pose, since that's
        algebraically guaranteed correct (A * (A^-1 * B) == B) -- was
        empirically found NOT to hold in this RoboDK setup (verified via
        diagnostic prints: the chip consistently jumped by very close to
        the TCP offset amount immediately on reparenting, even with that
        exact formula in place). Rather than trust an API contract that
        isn't holding here for reasons we couldn't fully pin down, this
        MEASURES the actual result after the standard attempt and, if
        it's off, applies a second correction computed directly from the
        observed discrepancy -- which fixes it regardless of the
        underlying cause."""
        target_world_pose = item.PoseAbs()
        item.setParentStatic(new_parent)
        item.setPose(new_parent.PoseAbs().inv() * target_world_pose)

        actual_world_pose = item.PoseAbs()
        if not self._pose_close(actual_world_pose, target_world_pose):
            # Standard formula didn't hold -- correct using the measured
            # error directly instead of the (apparently unreliable) formula.
            error = actual_world_pose.inv() * target_world_pose
            item.setPose(item.Pose() * error)

    # ---------- suction, simulated via reparenting the chip item ----------
    def suction_on(self, chip_item):
        """Unlike the real robot_controller (which just toggles a digital
        output), the simulated version needs to know *which* item to pick
        up, since there's no physical vacuum to sense a real object."""
        parent = self.tool if self.tool.Valid() else self.robot
        self._reparent_preserving_world_pose(chip_item, parent)
        self._held_item = chip_item

    def suction_off(self):
        if self._held_item is not None:
            self._reparent_preserving_world_pose(self._held_item, self.rdk.ActiveStation())
            self._held_item = None

    def suction_engaged(self, timeout_s=0.0):
        # No physical failure mode by default -- always succeeds once
        # suction_on() has attached an item. Inject random failures here
        # later if you want to stress-test sorting_task's retry logic.
        return self._held_item is not None

    def current_pose(self):
        pose_mat = self.robot.Pose()
        ur6_mm = robomath.Pose_2_UR(pose_mat)
        x, y, z, rx, ry, rz = ur6_mm
        return [x / 1000, y / 1000, z / 1000, rx, ry, rz]

    # ---------- ground-truth "perception" ----------
    def list_chip_items(self):
        """Returns every station object named 'chip_<color>_...' for a
        configured chip color, with its position and face-normal already
        expressed in the ROBOT BASE frame (matching what the real
        chip_detector.py + config.T_CAM_TO_BASE pipeline would hand back)."""
        chips = []
        base_pose_inv = self._robot_base_pose_abs.inv()

        for item in self.rdk.ItemList(ITEM_TYPE_OBJECT):
            name = item.Name().lower()
            for color in (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR):
                if name.startswith(f"chip_{color}"):
                    pose_in_base = base_pose_inv * item.PoseAbs()
                    pos_mm = pose_in_base.Pos()
                    position_m = np.array(pos_mm) / 1000.0
                    normal = np.array(pose_in_base.VZ())
                    normal = normal / np.linalg.norm(normal)
                    chips.append({
                        "item": item,
                        "color": color,
                        "position": position_m,
                        "normal": normal,
                    })
                    break
        return chips

    def shutdown(self):
        self.suction_off()
