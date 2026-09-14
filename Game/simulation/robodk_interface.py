"""
simulation/robodk_interface.py
Low-level bridge to an open RoboDK station: connection, motion, TCP
registration, and simulated suction (via item reparenting). This is the
shared foundation used by robodk_arm_publisher.py and robodk_camera.py --
neither of those talks to RoboDK directly, they both go through this.

Everything here reflects lessons learned the hard way while debugging an
earlier version of this exact bridge -- see the inline comments; several
of these are NOT what the RoboDK API docs would lead you to expect.
"""

import numpy as np
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_OBJECT
from robodk import robomath

import config

ROBODK_ROBOT_NAME = "UR3"
ROBODK_TOOL_NAME = "SuctionTool"
ROBODK_HOME_TARGET = "Home"        # Target item name; falls back to config.HOME_JOINTS if absent
BOARD_FRAME_NAME = "BoardFrame"    # optional; only relevant if you also use board-corner-rectangle placement


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

        # Push config.TOOL_TCP_OFFSET in as the robot's active TCP directly
        # -- config.py is the single source of truth, not whatever TCP the
        # tool item's own dialog in RoboDK might separately have set.
        self.robot.setPoseTool(self._ur6_to_robodk_pose(config.TOOL_TCP_OFFSET))

        # LESSON: Pose()/PoseAbs() on the ROBOT ITEM ITSELF return the
        # CURRENT forward-kinematics result (wherever the TCP is right
        # now), NOT the robot's fixed mounting location. The static base
        # is the PARENT item's absolute pose, cached once here since it
        # never changes during a run.
        self._robot_base_pose_abs = self.robot.Parent().PoseAbs()

        # LESSON: a robot's ACTIVE REFERENCE FRAME (what MoveL/MoveJ poses
        # are interpreted relative to) is a SEPARATE setting from the
        # station-tree parent used above. Without this, math and applied
        # motion can each be internally consistent yet still disagree.
        self.robot.setPoseFrame(self.robot.Parent())

        self.robot.setSpeed(config.DEFAULT_LIN_SPEED * 1000)      # RoboDK API takes mm/s
        self.robot.setAcceleration(config.DEFAULT_LIN_ACCEL * 1000)

        self._held_item = None

        # Enables RoboDK's own collision engine + lets you check
        # self.rdk.Collisions() after a move. Make sure chip objects are
        # excluded from colliding against BOTH the Robot and Tool items
        # in RoboDK's Collision Map (Tools > Collision Map), or every
        # intentional pick will register as a collision.
        self.rdk.setCollisionActive(1)

        # If a 'BoardFrame' item exists, OVERWRITE config.T_BOARD_TO_BASE
        # with its ground-truth pose in THIS simulation -- this is what
        # makes get_column_drop_pose() (and anything else using
        # T_BOARD_TO_BASE) correct regardless of where the board object
        # happens to sit in a given sim run, without needing simulation
        # positioning to exactly replicate the real setup. If no
        # BoardFrame item exists, config.T_BOARD_TO_BASE is left exactly
        # as calibrated in config.py -- this is also what happens
        # automatically on real hardware, since this whole file/class is
        # never touched there (ur3_interface.py doesn't import it).
        board_frame = self.rdk.Item(BOARD_FRAME_NAME)
        if board_frame.Valid():
            pose_in_base = self._robot_base_pose_abs.inv() * board_frame.PoseAbs()
            config.T_BOARD_TO_BASE = self._mat_to_transform_m(pose_in_base)
            print(f"[RoboDKInterface] '{BOARD_FRAME_NAME}' found -- "
                  f"config.T_BOARD_TO_BASE overwritten from simulation ground truth:")
            print(np.round(config.T_BOARD_TO_BASE, 4))
        else:
            print(f"[RoboDKInterface] No '{BOARD_FRAME_NAME}' item found in the station -- "
                  f"config.T_BOARD_TO_BASE left as whatever config.py already has "
                  f"(currently calibrated for your real setup, NOT this simulation).")

    # ---------- pose conversion ----------
    @staticmethod
    def _mat_to_transform_m(mat):
        """Converts a RoboDK Mat (position in mm) to a 4x4 numpy transform
        in meters."""
        T = np.eye(4)
        T[:3, 0] = mat.VX()
        T[:3, 1] = mat.VY()
        T[:3, 2] = mat.VZ()
        T[:3, 3] = np.array(mat.Pos()) / 1000.0
        return T

    def _ur6_to_robodk_pose(self, pose_ur6):
        """pose_ur6: [x, y, z, rx, ry, rz] in METERS/radians (UR/this
        project's convention). LESSON: robomath.UR_2_Pose() actually
        expects MILLIMETERS for translation despite mimicking UR's own
        pose format -- always scale m -> mm before calling it."""
        x, y, z, rx, ry, rz = pose_ur6
        return robomath.UR_2_Pose([x * 1000, y * 1000, z * 1000, rx, ry, rz])

    # ---------- motion ----------
    def go_home(self):
        target = self.rdk.Item(ROBODK_HOME_TARGET)
        if target.Valid():
            self.robot.MoveJ(target)
        else:
            self.robot.MoveJ([np.degrees(j) for j in config.HOME_JOINTS])

    def move_l(self, pose_ur6, slow=False):
        mat = self._ur6_to_robodk_pose(pose_ur6)
        speed = config.APPROACH_LIN_SPEED if slow else config.DEFAULT_LIN_SPEED
        self.robot.setSpeed(speed * 1000)
        self.robot.MoveL(mat)
        n_collisions = self.rdk.Collisions()
        if n_collisions > 0:
            print(f"WARNING: {n_collisions} collision pair(s) detected after this move -- "
                  f"check the 3D view and RoboDK's Collision Map.")

    def move_l_slow(self, pose_ur6):
        self.move_l(pose_ur6, slow=True)

    def current_pose(self):
        """Current TCP pose as a UR6 [x,y,z,rx,ry,rz], meters/radians."""
        ur6_mm = robomath.Pose_2_UR(self.robot.Pose())
        x, y, z, rx, ry, rz = ur6_mm
        return [x / 1000, y / 1000, z / 1000, rx, ry, rz]

    # ---------- suction, simulated via reparenting the chip item ----------
    @staticmethod
    def _pose_close(pose_a, pose_b, tol_mm=0.5):
        pa, pb = pose_a.Pos(), pose_b.Pos()
        return sum((a - b) ** 2 for a, b in zip(pa, pb)) ** 0.5 < tol_mm

    def _reparent_preserving_world_pose(self, item, new_parent):
        """Reparents `item` onto `new_parent`, preserving its exact world
        pose. LESSON: the textbook-correct formula (new_parent.PoseAbs().inv()
        * target_world_pose, algebraically guaranteed by A*(A^-1*B)=B) was
        empirically found to NOT reliably hold in RoboDK -- this measures
        the actual result and applies a second correction from the
        observed error if needed, which fixes it regardless of cause."""
        target_world_pose = item.PoseAbs()
        item.setParentStatic(new_parent)
        item.setPose(new_parent.PoseAbs().inv() * target_world_pose)

        actual_world_pose = item.PoseAbs()
        if not self._pose_close(actual_world_pose, target_world_pose):
            error = actual_world_pose.inv() * target_world_pose
            item.setPose(item.Pose() * error)

    def set_item_position_base(self, item, position_base):
        """Directly moves `item` to `position_base` (meters, robot base
        frame), preserving its CURRENT orientation and parent -- does
        NOT reparent it to anything. Used to instantly place a chip at
        its exact final cell position (simulating gravity) without
        physically dragging it or attaching it to the cell frame.

        Deliberately built ONLY from Mat operations already proven to
        work correctly elsewhere in this file (Pose_2_UR/UR_2_Pose,
        .Pos(), matrix multiply, .inv()) rather than raw row-indexed Mat
        editing, which hasn't been verified against this specific
        RoboDK installation. Also applies the same measure-and-correct
        pattern used for suction reparenting, for the same reason: don't
        trust pose composition to behave exactly as expected without
        checking the actual result."""
        parent = item.Parent()
        current_abs_ur6 = robomath.Pose_2_UR(item.PoseAbs())   # [x,y,z,rx,ry,rz], mm/rad

        target_pos_base_mm = np.array(position_base, dtype=float) * 1000.0
        target_abs_mat = self._robot_base_pose_abs * robomath.transl(*target_pos_base_mm)
        target_pos_mm = target_abs_mat.Pos()

        new_abs_ur6 = [target_pos_mm[0], target_pos_mm[1], target_pos_mm[2],
                       current_abs_ur6[3], current_abs_ur6[4], current_abs_ur6[5]]
        new_abs = robomath.UR_2_Pose(new_abs_ur6)
        item.setPose(parent.PoseAbs().inv() * new_abs)

        actual_pos_mm = np.array(item.PoseAbs().Pos())
        if np.linalg.norm(actual_pos_mm - target_pos_mm) > 0.5:
            error_mm = target_pos_mm - actual_pos_mm
            local_ur6 = robomath.Pose_2_UR(item.Pose())
            corrected_ur6 = [local_ur6[0] + error_mm[0], local_ur6[1] + error_mm[1],
                              local_ur6[2] + error_mm[2], local_ur6[3], local_ur6[4], local_ur6[5]]
            item.setPose(robomath.UR_2_Pose(corrected_ur6))

    def suction_on(self, chip_item):
        """Attaches chip_item to the tool (or robot, if no tool item is
        present). LESSON: call this only once the tool has actually
        reached CONTACT with the chip -- calling it any earlier (e.g. at
        an approach/hover height) will visibly teleport the chip up to
        the tool while still hovering, since simulated attachment is
        immediate, unlike a real vacuum pump building suction."""
        parent = self.tool if self.tool.Valid() else self.robot
        self._reparent_preserving_world_pose(chip_item, parent)
        self._held_item = chip_item

    def suction_off(self):
        if self._held_item is not None:
            self._reparent_preserving_world_pose(self._held_item, self.rdk.ActiveStation())
            self._held_item = None

    def suction_engaged(self, timeout_s=0.0):
        return self._held_item is not None

    # ---------- ground-truth chip listing ----------
    def list_chip_items(self):
        """Every station object named 'chip_<color>_...' for
        config.CHIP_ONE_COLOR / CHIP_TWO_COLOR, with position/normal in
        the ROBOT BASE frame."""
        chips = []
        base_pose_inv = self._robot_base_pose_abs.inv()

        for item in self.rdk.ItemList(ITEM_TYPE_OBJECT):
            name = item.Name().lower()
            for color in (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR):
                if name.startswith(f"chip_{color}"):
                    pose_in_base = base_pose_inv * item.PoseAbs()
                    position_m = np.array(pose_in_base.Pos()) / 1000.0
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
