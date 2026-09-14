"""
simulation/diagnose_alignment.py
Standalone diagnostic: prints every pose/transform involved in getting a
chip's position right, and moves the robot to hover (safely, above
contact) over one real chip so you can visually compare where the code
THINKS the chip is against where it actually sits in the 3D view.

Run this directly (not via sim_task.py) with the RoboDK station open:

    python -m simulation.diagnose_alignment

Read the printed numbers top to bottom -- each section tells you exactly
which piece of the pipeline to go check if something looks wrong.
"""

import numpy as np
from robodk import robomath

import config
from simulation.robodk_interface import RoboDKInterface
from planning.path_planner import build_pick_path


def print_mat(label, mat_mm):
    """mat_mm: a RoboDK Mat (mm). Prints position in both mm and m, plus
    the rotation part, for easy comparison against config.py values
    (which are all in meters)."""
    pos_mm = mat_mm.Pos()
    pos_m = [v / 1000.0 for v in pos_mm]
    print(f"{label}")
    print(f"    position (mm): {[round(v, 2) for v in pos_mm]}")
    print(f"    position (m):  {[round(v, 5) for v in pos_m]}")
    print(f"    VX: {[round(v, 4) for v in mat_mm.VX()]}")
    print(f"    VY: {[round(v, 4) for v in mat_mm.VY()]}")
    print(f"    VZ: {[round(v, 4) for v in mat_mm.VZ()]}")


def main():
    rdk_iface = RoboDKInterface()   # runs the exact same init as production code
    robot = rdk_iface.robot

    print("=" * 70)
    print("0. TOOL ATTACHMENT CHECK")
    print("=" * 70)
    print(f"Tool item valid: {rdk_iface.tool.Valid()}")
    if rdk_iface.tool.Valid():
        tool_parent = rdk_iface.tool.Parent()
        print(f"Tool's parent item: '{tool_parent.Name()}'")
        is_attached = (tool_parent.Name() == robot.Name())
        print(f"Tool is parented directly to the robot: {is_attached}")
        if not is_attached:
            print("    ^^^ THIS IS LIKELY THE BUG. If the tool mesh isn't parented")
            print("    to the robot itself, it stays fixed in place and never moves")
            print("    with the arm, no matter what the TCP math says. In RoboDK's")
            print("    Station Tree, drag the SuctionTool item directly onto the")
            print("    UR3 robot item (not onto UR3 BaseFrame or anywhere else).")
    print()

    print("=" * 70)
    print("1. ROBOT / TCP STATE")
    print("=" * 70)
    print(f"Robot item name: {robot.Name()}")
    print(f"Robot parent item name (used as the fixed base): {robot.Parent().Name()}")
    print()

    print("config.TOOL_TCP_OFFSET (what we INTENDED to set, meters):")
    print(f"    {config.TOOL_TCP_OFFSET}")
    print()

    print_mat("robot.PoseTool()  <- what RoboDK ACTUALLY has as the active TCP right now:",
              robot.PoseTool())
    print("    ^ Compare this to TOOL_TCP_OFFSET above (converted to mm).")
    print("      If these don't match, the TCP push in RoboDKInterface.__init__")
    print("      either isn't running, is being overridden afterward, or")
    print("      config.TOOL_TCP_OFFSET itself is still a placeholder (all zeros).")
    print()

    print_mat("robot.Parent().PoseAbs()  <- the fixed base transform we use:",
              rdk_iface._robot_base_pose_abs)
    print()

    print_mat("robot.PoseFrame()  <- what MoveL/MoveJ poses are ACTUALLY interpreted "
              "relative to:", robot.PoseFrame())
    print("    ^ This must match robot.Parent().PoseAbs() above exactly. If it")
    print("      doesn't, RoboDKInterface's setPoseFrame() call either isn't")
    print("      running or is being overridden somewhere after it.")
    print()

    print("=" * 70)
    print("2. BOARD TRANSFORM")
    print("=" * 70)
    print("config.T_BOARD_TO_BASE (after RoboDKInterface overwrote it from BoardFrame):")
    print(np.round(config.T_BOARD_TO_BASE, 5))
    print()

    print("=" * 70)
    print("3. ONE REAL CHIP, RAW vs COMPUTED")
    print("=" * 70)
    chips = rdk_iface.list_chip_items()
    if not chips:
        print("No chip_<color>_<n> items found in the station -- check naming.")
        return

    chip = chips[0]
    item = chip["item"]
    print(f"Using chip: {item.Name()}")
    print()
    print_mat("item.PoseAbs()  <- the chip's raw pose in the world, straight from RoboDK:",
              item.PoseAbs())
    print()
    print(f"Computed position in ROBOT BASE frame (what our code hands to path_planner):")
    print(f"    {chip['position']}  (meters)")
    print(f"Computed surface normal in ROBOT BASE frame:")
    print(f"    {chip['normal']}")
    print()

    print("=" * 70)
    print("4. LIVE VISUAL CHECK")
    print("=" * 70)
    print("Moving the robot to hover ABOVE this chip (approach height, no contact).")
    print("Watch the 3D view: does the tool tip end up directly above the chip,")
    print("or offset from it? The direction/distance of any offset you see tells")
    print("you exactly what's still wrong:")
    print("  - Offset straight down/up only       -> TOOL_TCP_OFFSET / TOOL_FLANGE_TO_CUP_TIP")
    print("  - Offset sideways, consistent amount  -> BoardFrame position, or T_CAM_TO_BASE")
    print("    on the real robot")
    print("  - Offset that changes with robot pose -> a remaining FK-dependent pose bug")
    print("    (shouldn't happen anymore, but if it does, tell me)")
    print()

    pick_path = build_pick_path(chip["position"], chip["normal"])
    approach_pose_ur6 = pick_path["approach"]
    target_mat = rdk_iface._ur6_to_robodk_pose(approach_pose_ur6)

    def _safe_joints():
        j = robot.Joints()
        try:
            # robot.Joints() returns a RoboDK column-vector Mat (rows of
            # single-element lists), not a plain list -- extract floats
            # directly from .rows rather than iterating j itself.
            return [round(float(row[0]), 1) for row in j.rows]
        except (TypeError, AttributeError, IndexError):
            return None   # comparison below just treats this as "unknown"

    joints_before = _safe_joints()
    rdk_iface.move_l(approach_pose_ur6)
    joints_after = _safe_joints()

    print("Intended target pose (mm), what we told the robot to reach:")
    print_mat("    ", target_mat)
    print()
    print("robot.Pose() AFTER the move -- where the robot actually thinks its")
    print("TCP ended up (relative to the active reference frame):")
    print_mat("    ", robot.Pose())
    print()
    print(f"Joints before move: {joints_before}")
    print(f"Joints after move:  {joints_after}")
    if joints_before is not None and joints_before == joints_after:
        print("    ^^^ Joints DID NOT CHANGE. The move was likely rejected")
        print("    (unreachable target / IK failure) rather than silently")
        print("    landing in the wrong place. Check RoboDK's own window for")
        print("    a red robot highlight or a popup warning.")
    print()
    print("Move sent. Check the 3D view now.")


if __name__ == "__main__":
    main()
