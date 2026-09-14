"""
simulation/robodk_game.py
Runs the complete Connect Four game (board.py + ai.py + game_manager.py,
all UNMODIFIED) against a RoboDK simulation instead of real hardware.

Station requirements:
  - Robot named to match ROBODK_ROBOT_NAME in robodk_interface.py (default "UR3")
  - Suction tool attached to the flange, named to match ROBODK_TOOL_NAME ("SuctionTool")
  - A board object (any name -- not read by name, only its chip and cell
    children matter)
  - Chip objects named "chip_<color>_<n>", e.g. "chip_red_1", "chip_yellow_1"
  - 42 Frame items named "Cell_R<row>_C<col>" for row 0-5, col 0-6,
    positioned exactly where a chip resting in that board cell should sit
  - Optionally a Target item named "Home" for the resting pose

Turn flow:
  - Robot's turn: AI picks a column, the arm picks up one of its own
    color's chips and drops it at that column's entrance, then the chip
    is placed programmatically into its exact final cell -- no manual
    dragging needed for robot moves.
  - Player's turn: you drag one of your own chips into its target cell
    directly, then press Enter at the prompt. game_manager.py's own
    change-detection (via read_grid()) picks up whatever you did and
    validates it -- fouls (multi-move, floating chips, wrong color) are
    caught exactly the same way they would be with a real camera.
"""

from board import PLAYER, ROBOT, fog_of_war_cells
from ai import RobotAI
from game_manager import GameManager
import config

from simulation.robodk_interface import RoboDKInterface
from simulation.robodk_arm_publisher import RoboDKSuctionArmPublisher
from simulation.robodk_camera import RoboDKCellCamera


def _prompt_for_color():
    options = (config.CHIP_ONE_COLOR, config.CHIP_TWO_COLOR)
    while True:
        choice = input(f"Which color is the player using? {options}: ").strip().lower()
        if choice in options:
            return choice
        print(f"'{choice}' isn't one of {options}, try again.")


def _player_move_via_manual_drag(baseline_grid):
    """Instead of typing a column, the player physically drags a chip
    into place inside RoboDK, then confirms here by pressing Enter.
    game_manager.py's _take_player_turn() then falls through to
    _wait_for_camera_change(), which re-reads the REAL current grid and
    validates whatever actually changed -- this function's return value
    (a column number) is a required part of the ArmPublisher/game_manager
    interface contract but is NOT otherwise used downstream here, since
    our camera has no force_set_grid() to short-circuit into (that path
    only exists for the mock/testing camera). Any currently-open column
    is a safe placeholder value."""
    input("\nYour turn: drag one of your chips into its target cell in "
          "RoboDK, then press Enter here to continue...")
    reachable = fog_of_war_cells(baseline_grid)
    if not reachable:
        raise RuntimeError("Board is full; no columns available.")
    return next(iter(reachable.keys()))


def _make_robot_place_chip_fn(arm, camera):
    """Returns a callback(row, col) that instantly moves the chip the
    robot just released to its exact final cell position -- removing the
    need for a human to drag it down precisely. Uses arm.last_placed_chip_item
    (set right before suction_off() in pick_and_place_to_pose) so there's
    no ambiguity about which chip to move, and camera.cell_positions for
    the exact target, so it's tied to the same geometry used for grid
    reading -- no separate hardcoded numbers involved."""
    def place_chip(row, col):
        chip_item = arm.last_placed_chip_item
        if chip_item is None:
            print(f"Warning: no chip tracked for the last move -- can't "
                  f"auto-place into R{row}C{col}. Falling back to manual drag.")
            return
        target_pos = camera.cell_positions.get((row, col))
        if target_pos is None:
            print(f"Warning: no Cell_R{row}_C{col} frame found -- can't "
                  f"auto-place. Falling back to manual drag.")
            return
        arm.rdk_iface.set_item_position_base(chip_item, target_pos)
        print(f"Chip placed automatically into R{row}C{col}.")
    return place_chip


def main():
    player_color = _prompt_for_color()
    robot_color = (config.CHIP_TWO_COLOR if player_color == config.CHIP_ONE_COLOR
                    else config.CHIP_ONE_COLOR)

    # Both the arm publisher and camera share ONE RoboDK connection --
    # the arm publisher owns it (does TCP/collision/base-frame setup),
    # the camera just borrows it for read-only queries.
    arm = RoboDKSuctionArmPublisher()
    camera = RoboDKCellCamera(arm.rdk_iface, player_color=player_color, robot_color=robot_color)

    # By default, column positions come from config.COLUMN_POSITIONS
    # (board-local, measured from the real board) transformed through
    # config.T_BOARD_TO_BASE -- correct in both real and simulated
    # contexts AS LONG AS your station has a 'BoardFrame' item so
    # T_BOARD_TO_BASE reflects where THIS board actually is (see
    # robodk_arm_publisher.py's _get_column_drop_pose docstring).
    #
    # Uncomment the line below to instead derive column positions from
    # the Cell_R0_C<col> frames directly -- useful as a one-time
    # cross-check: run once with it enabled, once without, and compare
    # the printed drop poses. A large disagreement between the two means
    # your BoardFrame item's pose doesn't match reality yet.
    #
    # arm.column_position_provider = camera.get_column_entrance_position

    ai = RobotAI(depth=5)
    manager = GameManager(
        camera=camera,
        arm_publisher=arm,
        ai=ai,
        poll_interval=0.3,
        # Robot turns place the chip programmatically (see
        # robot_place_chip_fn below) so they resolve immediately -- only
        # the PLAYER's turn still involves a human dragging a chip, so
        # only that side needs an unbounded wait.
        robot_move_timeout=30.0,
        player_move_timeout=None,
        player_move_input_fn=_player_move_via_manual_drag,
        robot_place_chip_fn=_make_robot_place_chip_fn(arm, camera),
        verbose=True,
        stable_checks=2,
        robot_settle_time=1.0,
        player_color=player_color,
        robot_color=robot_color,
    )

    try:
        manager.run()
    finally:
        arm.shutdown()


if __name__ == "__main__":
    main()
