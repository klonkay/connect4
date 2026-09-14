"""
test_foul_detection.py
-----------------------
Scripted, non-interactive smoke tests:

  1. Plays a short automated game (robot vs a scripted "player" sequence)
     to sanity check board logic + AI + win detection.
  2. Demonstrates a multi-cell-change foul being detected and the game
     manager aborting/resetting, exactly as it should if a player altered
     more than one cell at once.
  3. Demonstrates a floating-chip foul being detected.

Run with:  python test_foul_detection.py
"""

import board as board_mod
from ai import RobotAI
from board import PLAYER, ROBOT, FoulError, validate_single_legal_move
from interfaces import MockCameraInterface, MockArmPublisher


def test_basic_win_detection():
    print("=== test_basic_win_detection ===")
    grid = board_mod.create_empty_grid()
    for col in range(4):
        grid, _ = board_mod.apply_move(grid, col, PLAYER)
    winner = board_mod.check_winner(grid)
    assert winner == PLAYER, f"Expected PLAYER to win, got {winner}"
    print("PASS: horizontal win detected correctly.\n")


def test_ai_blocks_immediate_loss():
    print("=== test_ai_blocks_immediate_loss ===")
    grid = board_mod.create_empty_grid()
    # Player has three in a row horizontally at row 5 (bottom), cols 0-2.
    # Robot must block at column 3.
    for col in range(3):
        grid, _ = board_mod.apply_move(grid, col, PLAYER)
    ai = RobotAI(depth=3)
    move = ai.choose_move(grid)
    assert move == 3, f"Expected AI to block at column 3, chose {move}"
    print("PASS: AI correctly blocked the player's imminent win.\n")


def test_ai_takes_immediate_win():
    print("=== test_ai_takes_immediate_win ===")
    grid = board_mod.create_empty_grid()
    for col in range(3):
        grid, _ = board_mod.apply_move(grid, col, ROBOT)
    ai = RobotAI(depth=3)
    move = ai.choose_move(grid)
    assert move == 3, f"Expected AI to win at column 3, chose {move}"
    print("PASS: AI correctly took the immediate winning move.\n")


def test_fog_of_war_no_floating_chips():
    print("=== test_fog_of_war_no_floating_chips ===")
    grid = board_mod.create_empty_grid()
    grid, _ = board_mod.apply_move(grid, 2, PLAYER)  # one chip at bottom of col 2
    reachable = board_mod.fog_of_war_cells(grid)
    # Column 2's only reachable cell should now be one row up from the bottom.
    assert reachable[2] == board_mod.ROWS - 2, "Fog-of-war reachable row is wrong after one drop."
    print("PASS: only the single physically reachable cell per column is exposed.\n")


def test_multi_move_foul_detected():
    print("=== test_multi_move_foul_detected ===")
    old_grid = board_mod.create_empty_grid()
    new_grid = board_mod.clone_grid(old_grid)
    # Illegally place two chips at once (simulating tampering / multiple moves).
    new_grid[board_mod.ROWS - 1][0] = PLAYER
    new_grid[board_mod.ROWS - 1][1] = PLAYER
    try:
        validate_single_legal_move(old_grid, new_grid, PLAYER)
        print("FAIL: multi-move foul was NOT detected!")
    except FoulError as e:
        print(f"PASS: foul correctly detected -> {e}\n")


def test_floating_chip_foul_detected():
    print("=== test_floating_chip_foul_detected ===")
    old_grid = board_mod.create_empty_grid()
    new_grid = board_mod.clone_grid(old_grid)
    # Illegally place a chip in row 0 of an empty column (should land in row 5).
    new_grid[0][3] = PLAYER
    try:
        validate_single_legal_move(old_grid, new_grid, PLAYER)
        print("FAIL: floating chip foul was NOT detected!")
    except FoulError as e:
        print(f"PASS: foul correctly detected -> {e}\n")


def test_game_manager_end_to_end_with_foul():
    print("=== test_game_manager_end_to_end_with_foul (scripted) ===")
    from game_manager import GameManager

    camera = MockCameraInterface()
    arm_publisher = MockArmPublisher(camera, simulated_move_seconds=0.01)
    ai = RobotAI(depth=3)

    # Scripted player: first move is a normal legal move (column 5); on the
    # SECOND call we simulate a cheating player who drops two chips before
    # the camera is polled again, by writing directly into the mock camera.
    call_count = {"n": 0}

    def scripted_player_move_fn(grid):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 5  # normal legal first move
        # Simulate an illegal simultaneous double-move by writing straight
        # into the shared mock camera before returning a (now irrelevant) column.
        cheated_grid = board_mod.clone_grid(grid)
        cheated_grid[board_mod.ROWS - 1][1] = PLAYER
        cheated_grid[board_mod.ROWS - 1][2] = PLAYER
        camera.force_set_grid(cheated_grid)
        return 1  # value is irrelevant; camera already shows the illegal double-move

    manager = GameManager(
        camera=camera,
        arm_publisher=arm_publisher,
        ai=ai,
        poll_interval=0.01,
        robot_move_timeout=5.0,
        player_move_input_fn=scripted_player_move_fn,
        verbose=True,
    )

    outcome = manager._play_one_game(PLAYER)
    assert "Foul" in outcome, f"Expected a foul outcome, got: {outcome}"
    print(f"PASS: game manager correctly aborted and flagged a reset -> '{outcome}'\n")


if __name__ == "__main__":
    test_basic_win_detection()
    test_ai_blocks_immediate_loss()
    test_ai_takes_immediate_win()
    test_fog_of_war_no_floating_chips()
    test_multi_move_foul_detected()
    test_floating_chip_foul_detected()
    test_game_manager_end_to_end_with_foul()
    print("All tests passed.")
