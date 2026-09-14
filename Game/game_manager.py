"""
game_manager.py
----------------
Orchestrates a full game. This is the "referee": it never trusts stated
intentions, only the grid state actually observed from the camera.

Core loop philosophy (matches the physical system being modeled):
  - The camera is polled for the CURRENT grid at every step.
  - When it's the human's turn, we simply wait until the camera reports a
    new grid state (the human physically drops their own chip).
  - When it's the robot's turn, we compute a move, command the arm to
    pick and place, then wait for the camera to confirm the completed move.
  - The grid is validated against a single legal move (foul detection).
  - Polling can be slowed down and change detection requires stability.
"""

import time
import board
from board import (
    ROWS, COLS, EMPTY, PLAYER, ROBOT, FoulError,
    check_winner, is_full, validate_single_legal_move, render,
)
from ai import RobotAI
from interfaces import ArmMotionError
import config

class GameManager:
    def __init__(self, camera, arm_publisher, ai=None,
                 poll_interval=0.5, robot_move_timeout=60.0,
                 player_move_timeout=None,
                 player_move_input_fn=None,
                 robot_move_confirm_fn=None,
                 robot_place_chip_fn=None,
                 verbose=True,
                 stable_checks=2,
                 robot_settle_time=2.0,
                 player_color=None,
                 robot_color=None):
        """
        camera:                object implementing CameraInterface.read_grid()
        arm_publisher:         object implementing ArmPublisher
        ai:                    RobotAI instance (created with defaults if None)
        poll_interval:         seconds between camera polls while waiting for a move
        robot_move_timeout:    seconds to wait for the arm to complete a move before
                               giving up (None = wait forever)
        player_move_timeout:   seconds to wait for the human to drop a chip before
                               giving up (None = wait forever)
        player_move_input_fn:  OPTIONAL callback used only in simulation/testing to
                               stand in for a human physically dropping a chip.
        robot_move_confirm_fn: OPTIONAL zero-argument callback called right after
                               the robot's physical move completes, BEFORE the
                               settle-sleep and camera polling begin. Currently
                               unused by default -- superseded by
                               robot_place_chip_fn below for simulation, which
                               places the chip programmatically instead of
                               needing a manual pause. Leave as None for real
                               hardware, where gravity happens automatically.
        robot_place_chip_fn:   OPTIONAL callback(row, col) called right after
                               the robot's physical move completes (before
                               robot_move_confirm_fn, settle-sleep, and
                               polling). Intended for simulation: since the
                               robot only drops a chip at a column's entrance
                               (gravity isn't simulated), this lets the
                               simulation bridge instantly move that exact
                               chip to its correct final cell position,
                               removing the need for a human to drag it down
                               precisely -- which was error-prone enough to
                               cause silent "stuck" games when a dragged chip
                               matched the wrong cell. Leave as None for real
                               hardware, where gravity places the chip itself.
        verbose:               print logs
        stable_checks:         number of consecutive identical grid reads required
                               before accepting a change (reduces false positives)
        robot_settle_time:     seconds to wait after robot motion before polling
                               (to allow the arm to clear the view)
        player_color:          the chip color string the human is using (e.g. "red").
                               REQUIRED for _take_robot_turn to reliably find a chip
                               of the robot's own color -- without it, this class
                               previously guessed based on a hardcoded assumption
                               that ROBOT (the fixed board value 2) always meant
                               config.CHIP_TWO_COLOR, which is wrong whenever the
                               player picks that same color for themselves.
        robot_color:           the chip color string the robot is using. If both
                               player_color and robot_color are omitted, falls back
                               to the old hardcoded guess (with a printed warning),
                               so existing callers that don't care about color
                               (e.g. the mock/simulate-only mode) keep working.
        """
        self.camera = camera
        self.arm_publisher = arm_publisher
        self.ai = ai or RobotAI()
        self.poll_interval = poll_interval
        self.robot_move_timeout = robot_move_timeout
        self.player_move_timeout = player_move_timeout
        self.player_move_input_fn = player_move_input_fn
        self.robot_move_confirm_fn = robot_move_confirm_fn
        self.robot_place_chip_fn = robot_place_chip_fn
        self.verbose = verbose
        self.stable_checks = stable_checks
        self.robot_settle_time = robot_settle_time
        self.player_color = player_color
        self.robot_color = robot_color

    # ---------------------------------------------------------------
    # top level
    # ---------------------------------------------------------------

    def run(self):
        """Run games back-to-back until the user chooses to stop."""
        while True:
            first_player = self._ask_who_goes_first()
            outcome = self._play_one_game(first_player)
            self._log(f"\n=== Game ended: {outcome} ===\n")
            if not self._ask_play_again():
                self._log("Goodbye!")
                return

    def _ask_who_goes_first(self):
        while True:
            choice = input("Who goes first? [p]layer / [r]obot: ").strip().lower()
            if choice in ("p", "player"):
                return PLAYER
            if choice in ("r", "robot"):
                return ROBOT
            print("Please type 'p' or 'r'.")

    def _ask_play_again(self):
        choice = input("Play again? [y/n]: ").strip().lower()
        return choice.startswith("y")

    # ---------------------------------------------------------------
    # one game
    # ---------------------------------------------------------------

    def _play_one_game(self, first_player):
        self._reset_physical_board_prompt()
        turn = first_player
        current_grid = self.camera.read_grid()

        if any(cell != EMPTY for row in current_grid for cell in row):
            self._log("Warning: board is not empty at game start according to the camera. "
                       "Proceeding anyway - make sure the physical board is actually clear.")

        self._log("Starting game. Board:\n" + render(current_grid))

        while True:
            if check_winner(current_grid) is not None:
                winner = check_winner(current_grid)
                return "Player wins!" if winner == PLAYER else "Robot wins!"
            if is_full(current_grid):
                return "Draw!"

            baseline = current_grid

            try:
                if turn == PLAYER:
                    new_grid = self._take_player_turn(baseline)
                else:
                    new_grid = self._take_robot_turn(baseline)

                row, col = validate_single_legal_move(baseline, new_grid, turn)
                self._log(f"{'Player' if turn == PLAYER else 'Robot'} played column {col}.")
                self._log(render(new_grid))

            except FoulError as e:
                self._log(f"\n*** FOUL DETECTED: {e} ***")
                self._log("*** Game aborted. Resetting board. ***\n")
                return f"Foul by {'player' if turn == PLAYER else 'robot/system'} " \
                       f"turn (reset triggered): {e}"

            except TimeoutError as e:
                self._log(f"\n*** TIMEOUT: {e} ***")
                self._log("*** Game aborted. Resetting board. ***\n")
                return f"Timeout waiting for move (reset triggered): {e}"

            except ArmMotionError as e:
                self._log(f"\n*** ROBOT MOTION FAILURE: {e} ***")
                self._log("*** Game aborted. Resetting board. ***\n")
                return f"Robot motion failure (reset triggered): {e}"

            current_grid = new_grid
            turn = ROBOT if turn == PLAYER else PLAYER

    # ---------------------------------------------------------------
    # turns
    # ---------------------------------------------------------------

    def _take_player_turn(self, baseline_grid):
        self._log("Player's turn. Waiting for a chip to be dropped...")

        if self.player_move_input_fn is not None:
            # Simulation/testing path: caller supplies the column, we "physically"
            # drop it by writing directly into the mock camera.
            col = self.player_move_input_fn(baseline_grid)
            row = board.next_open_row(baseline_grid, col)
            if row is None:
                raise FoulError(f"Column {col} is full; illegal move attempted.")
            simulated_grid, _ = board.apply_move(baseline_grid, col, PLAYER)
            if hasattr(self.camera, "force_set_grid"):
                self.camera.force_set_grid(simulated_grid)

        return self._wait_for_camera_change(baseline_grid, timeout=self.player_move_timeout)

    def _take_robot_turn(self, baseline_grid, max_pickup_retries=3):
        """
        There's no physical sensor confirming a successful chip pickup
        (no suction-pressure switch, nothing on the tool itself) -- a
        failed grip is only detectable INDIRECTLY: the arm completes its
        full move sequence regardless of whether it's actually holding a
        chip, so if the board shows NO change at all afterward, that's
        strong evidence the pickup itself failed, not that something
        deeper is wrong. Rather than letting that surface as an
        immediate timeout/game-ending failure, this retries the
        pickup-and-play attempt up to max_pickup_retries times -- each
        retry re-reads the chip's position fresh from the camera (rather
        than reusing a stale pose), since a chip that failed to be
        picked up should still be roughly where it was.
        """
        self._log("Robot's turn. Evaluating board...")
        col = self.ai.choose_move(baseline_grid)
        row = board.next_open_row(baseline_grid, col)
        self._log(f"Robot decided on column {col} (row {row}).")

        for attempt in range(1, max_pickup_retries + 1):
            # ---- Get a chip of the robot's colour from the camera --
            # re-read FRESH every attempt.
            pickup_pose = None
            if hasattr(self.camera, 'get_chip_positions'):
                try:
                    if hasattr(self.camera, 'update'):
                        self.camera.update()
                    all_chips = self.camera.get_chip_positions()
                    if self.robot_color is not None:
                        robot_color_name = self.robot_color
                    else:
                        # Fallback ONLY -- this guess is wrong whenever the
                        # player happens to pick config.CHIP_TWO_COLOR for
                        # themselves, since it hardcodes ROBOT==2 ->
                        # CHIP_TWO_COLOR regardless of the actual choice made
                        # at game start. Always pass player_color/robot_color
                        # into GameManager's constructor to avoid this path.
                        self._log("Warning: GameManager was constructed without "
                                  "robot_color -- guessing based on a fixed "
                                  "assumption that may not match the player's "
                                  "actual color choice.")
                        robot_color_name = config.CHIP_TWO_COLOR if ROBOT == 2 else config.CHIP_ONE_COLOR
                    # NOTE: chips are Chip OBJECTS (attribute access: .color,
                    # .position_m), as actually returned by
                    # CombinedCamera.get_chip_positions() -- not dicts. Dict-
                    # style access here would raise an AttributeError the
                    # first time this runs against a real camera.
                    robot_chips = [c for c in all_chips if c.color == robot_color_name]
                    if robot_chips:
                        chip = robot_chips[0]   # you can improve selection later
                        x, y, z = chip.position_m
                        pickup_pose = [x, y, z, 2.905, 1.199, 0.0]
                        self._log(f"Found robot chip at x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    else:
                        self._log("No robot-coloured chips found on the board.")
                except Exception as e:
                    self._log(f"Error getting chip positions: {e}")

            # ---- Fallback to fixed pickup ----
            if pickup_pose is None:
                self._log("Falling back to fixed pickup position.")
                fixed_xyz = self.arm_publisher.pickup_positions.get(ROBOT)
                if fixed_xyz is None:
                    raise RuntimeError("No fixed pickup position for robot chip.")
                pickup_pose = fixed_xyz + [2.905, 1.199, 0.0]

            # ---- Execute the move -- use the catcher-based sequence when
            # the arm publisher supports it (real hardware), falling back to
            # the plain pick-and-place for simulation/mock arm publishers
            # that don't implement play_column_move.
            self._log(f"Picking chip and placing into column (attempt {attempt}/{max_pickup_retries}).")
            if hasattr(self.arm_publisher, 'play_column_move'):
                self.arm_publisher.play_column_move(pickup_pose, col)
            else:
                drop_pose = self.arm_publisher._get_column_drop_pose(col, approach=False, orientation='game')
                self.arm_publisher.pick_and_place_to_pose(pickup_pose, drop_pose)

            # ---- Simulation: instantly place the chip at its exact final
            # cell position, instead of leaving it at the column entrance
            # for a human to drag down. Removes the imprecise manual step
            # that could cause a dragged chip to register as the wrong cell
            # (or the same cell as an already-placed chip, which silently
            # looks like "no change" to the grid and hangs polling forever).
            if self.robot_place_chip_fn is not None:
                self.robot_place_chip_fn(row, col)

            # ---- Give the caller a chance to intervene BEFORE polling starts.
            # Without this, in simulation, the just-dropped chip can already
            # be close enough to a cell frame (e.g. the column's top cell) to
            # get picked up and validated by the very next poll, before a
            # human has any real chance to drag it down to simulate gravity --
            # which reads as a "floating chip" foul even though nobody
            # actually placed it there on purpose.
            if self.robot_move_confirm_fn is not None:
                self.robot_move_confirm_fn()

            # ---- Wait for the robot to settle and then for camera change ----
            self._log(f"Waiting {self.robot_settle_time}s for robot to clear the view...")
            time.sleep(self.robot_settle_time)

            try:
                return self._wait_for_camera_change(baseline_grid, timeout=self.robot_move_timeout)
            except TimeoutError:
                if attempt < max_pickup_retries:
                    self._log(f"No board change detected after attempt {attempt}/"
                              f"{max_pickup_retries} -- with no suction sensor to "
                              f"confirm a successful grip, this most likely means "
                              f"the chip was never actually picked up. Retrying "
                              f"with a freshly re-read chip position...")
                    continue
                else:
                    self._log(f"No board change detected after {max_pickup_retries} "
                              f"attempts. Giving up on this turn.")
                    raise

    # ---------------------------------------------------------------
    # low-level camera polling with stability check
    # ---------------------------------------------------------------

    def _wait_for_camera_change(self, baseline_grid, timeout=None, heartbeat_interval=3.0):
        """
        Poll the camera until the grid changes from the baseline.
        Uses stability: the new grid must be the same for `stable_checks` consecutive reads.

        Prints a heartbeat every `heartbeat_interval` seconds showing the
        currently observed grid -- without this, a genuine "nothing is
        matching yet" situation (e.g. a drag that's just outside
        CELL_MATCH_TOL_M) looks EXACTLY like a frozen program, especially
        with timeout=None (wait forever). This makes that distinction
        visible instead of silent.
        """
        start = time.time()
        last_seen_grid = None
        consecutive = 0
        last_heartbeat = start

        while True:
            observed = self.camera.read_grid()

            if observed != baseline_grid:
                # Potential change detected – check stability
                if observed == last_seen_grid:
                    consecutive += 1
                else:
                    # New candidate – reset counter
                    consecutive = 1
                    last_seen_grid = observed

                if consecutive >= self.stable_checks:
                    # Change is stable – return the grid
                    return observed
            else:
                # No change – reset stability counter
                consecutive = 0
                last_seen_grid = None

            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                if observed == baseline_grid:
                    self._log(f"[still waiting] No change detected yet "
                               f"({now - start:.0f}s elapsed). If you've already "
                               f"moved a chip, it isn't registering as being in "
                               f"any cell -- check CELL_MATCH_TOL_M / the chip's "
                               f"actual position with simulation.diagnose_grid.")
                else:
                    self._log(f"[still waiting] Detected a change but it hasn't "
                               f"been stable yet ({consecutive}/{self.stable_checks} "
                               f"consistent reads, {now - start:.0f}s elapsed).")

            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"No stable move detected within {timeout} seconds."
                )

            time.sleep(self.poll_interval)

    # ---------------------------------------------------------------
    # misc
    # ---------------------------------------------------------------

    def _reset_physical_board_prompt(self):
        if hasattr(self.camera, "reset"):
            self.camera.reset()
        else:
            input("Please clear the physical board, then press Enter to begin...")

    def _log(self, msg):
        if self.verbose:
            print(msg)