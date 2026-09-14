# UR3 Connect 4 & Chip Sorting

A UR3-based system that plays Connect 4 against a human opponent using a minimax + alpha-beta AI, tracks the
board via a fixed RealSense camera, and can also sort chips by color independently of the game. This README
covers this `Game/` folder specifically — see the [repository root README](../README.md) if you're not sure this
is the project you're looking for.

> **Physical setup note:** the transforms and poses committed in `config.py` (`T_CAM_TO_BASE`,
> `T_BOARD_TO_BASE`, `CATCHER_DROP_POSE`, `COLUMN_POSITIONS`, etc.) were measured for one specific physical rig.
> If you're setting this up on different hardware or a different board, you'll need to re-run the calibration
> steps below and re-measure the catcher/column positions for your own setup.

---

## Table of contents

- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Project structure](#project-structure)
- [A path issue to fix before running calibration](#a-path-issue-to-fix-before-running-calibration)
- [Calibration](#calibration)
- [Running the project](#running-the-project)
  - [Full CLI reference](#full-cli-reference)
  - [Playing a real game](#playing-a-real-game)
  - [Sorting chips by color](#sorting-chips-by-color)
  - [Simulation](#simulation)
  - [Automated tests](#automated-tests)
  - [Utility / debug scripts](#utility--debug-scripts)
- [Configuration reference (`config.py`)](#configuration-reference-configpy)
- [Things worth checking / known rough edges](#things-worth-checking--known-rough-edges)

---

## Hardware requirements

- Universal Robots **UR3** (CB3 controller), reachable over the network, with a suction end effector wired to a
  digital output pin.
- Intel RealSense **D435** (or compatible), fixed rigidly overlooking the board — **not** mounted on the arm
  (Eye-to-Hand configuration).
- A Connect 4 board and single-color chip set, plus the chip-alignment "catcher" fixture used during pickup.
- An ArUco marker for calibration (mounted temporarily on the robot flange), and a second one permanently
  attached to the board.

## Software requirements

```bash
pip install opencv-contrib-python numpy pyrealsense2 urx
```

There is no `requirements.txt` committed in this folder — the line above covers everything needed for the real
hardware path. `board.py` and `ai.py` only need `numpy` and the standard library, so game-logic changes can be
tested without the rest of the stack installed.

For the RoboDK simulation path (`simulation/`), you additionally need RoboDK itself installed, with its Python
API importable (`pip install robodk`, or use the copy RoboDK ships with the application).

> `urx` may need a small compatibility shim on newer Python versions:
> ```python
> import collections, collections.abc
> if not hasattr(collections, "Iterable"):
>     collections.Iterable = collections.abc.Iterable
> ```
> This is already included at the top of `calibration/calibrate_eye_to_hand.py`; add it wherever else you
> `import urx` directly if you hit `AttributeError: module 'collections' has no attribute 'Iterable'`.

## Project structure

```
Game/
├── main_windows.py            # Entry point — parses CLI flags, dispatches to a mode below
├── config.py                  # Single source of truth for tunable values (see reference table below)
├── game_manager.py            # GameManager — orchestrates a match: turns, foul detection, retries
├── board.py                   # Pure Connect 4 rules — 6x7 grid, win check, gravity — no hardware dependency
├── ai.py                      # RobotAI — minimax + alpha-beta move selection
├── interfaces.py              # Abstract CameraInterface / ArmPublisher + Mock implementations
├── combined_camera.py         # CombinedCamera — chip detection AND board-state reading (RealSense)
├── ur3_interface.py           # UR3SuctionArmPublisher — real robot control via urx, incl. catcher sequence
├── sorting_task.py            # SortingTask — chip-sorting state machine, independent of the game
├── sorting_config.py          # Sorting-specific configuration
├── test_foul_detection.py     # Scripted tests, run via `--test` (simulation-backed)
├── depth_test.py              # Standalone: click a pixel on the live feed, print the depth there
├── live_grid_detection.py     # Standalone: click board corners on a live feed, preview grid classification live
├── my_preset.json             # Optional RealSense "advanced mode" preset (off by default — see below)
│
├── calibration/
│   ├── calibrate_eye_to_hand.py   # Camera <-> robot base calibration — run this FIRST
│   └── board_calibration.py       # Board <-> robot base calibration — run this SECOND
│
└── simulation/
    ├── robodk_interface.py
    ├── robodk_arm_publisher.py
    ├── robodk_camera.py
    └── robodk_game.py
```

## A path issue to fix before running calibration

`calibration/board_calibration.py` imports `config` directly:
```python
from config import T_CAM_TO_BASE
```
but `config.py` lives one directory up, in `Game/`, not in `Game/calibration/`. Running it as
`python calibration/board_calibration.py` (or `cd`-ing into `calibration/` first) will fail with
`ModuleNotFoundError: No module named 'config'`, because Python adds the *script's own* directory to the import
path, not the current working directory or its parent.

Until this is restructured, the reliable way to run it is to point Python at `Game/` explicitly, e.g. from the
`Game/` directory:

```bash
# Windows (cmd)
set PYTHONPATH=%cd%
python calibration\board_calibration.py

# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
python calibration\board_calibration.py

# Linux/macOS
PYTHONPATH=. python calibration/board_calibration.py
```

`calibration/calibrate_eye_to_hand.py` does **not** have this problem — it keeps its own `ROBOT_IP` and settings
self-contained at the top of the file rather than importing from `config.py`, so it runs directly with no
`PYTHONPATH` adjustment needed.

## Calibration

Run these **in order**, once per physical setup:

### 1. Camera → robot base

```bash
cd Game
python calibration/calibrate_eye_to_hand.py
```
- Print an ArUco marker (`DICT_5X5_50`) and mount it rigidly on the robot's flange.
- Put the robot in **freedrive**, move to a new pose, press Enter to capture. Repeat for 15–25 poses, varying
  both position *and* orientation.
- Each capture shows a reprojection-error number alongside the preview — use that, not just the picture, to
  decide keep/discard.
- Writes `cam_to_base_transform.json`. Copy the matrix into `config.T_CAM_TO_BASE`.

### 2. Board → robot base

```bash
# see "A path issue" above for the PYTHONPATH requirement
python calibration/board_calibration.py
```
- Uses a **second**, separate ArUco marker, permanently attached to the board.
- Averages ~15 frames (discarding any above `MAX_REPROJ_ERROR_PX`) into a single board pose, combined with
  `T_CAM_TO_BASE` to produce `T_BOARD_TO_BASE`. Copy the result into `config.T_BOARD_TO_BASE`.

There is no separate `validate_calibration.py` / `calibrate_intrinsics.py` / `check_undistortion.py` currently
committed in this folder — if you're relying on notes that mention those, they haven't been added to this repo
yet.

## Running the project

Everything below (except the RoboDK simulation, which has its own entry point) goes through `main_windows.py`.

### Full CLI reference

```
--simulate          Run in simulation mode (no hardware, terminal play)
--camera-3d         Use RealSense 3D camera (instead of the 2D webcam fallback)
--robot-ip IP       UR3 IP address (default: 192.168.40.27)
--camera-index N    Webcam index, only relevant without --camera-3d (default: 0)
--suction-pin N     Digital output pin for suction (default: 1)
--calibrate         Passed through to real mode, but currently only prints a message —
                     the underlying calibration methods aren't implemented yet (see below)
--test              Run the scripted test suite and exit
--sort              Run the sorting task only, then optionally start a game after
```
Priority when multiple flags are given: `--sort` is checked first, then `--test`, then `--simulate`; otherwise
real hardware mode runs.

### Playing a real game

```bash
python main_windows.py --camera-3d
```
- On startup: `CombinedCamera` initializes (one-shot board-edge detection + warm-up), then the robot connects,
  sets its TCP offset, and homes.
- You'll be prompted for your chip color and then the game runs, alternating turns automatically with foul
  detection after every move.
- Omitting `--camera-3d` falls back to a 2D webcam, which **cannot** do real chip/board detection — only useful
  for exercising the non-vision code paths.

### Sorting chips by color

```bash
python main_windows.py --sort
```
Re-reads the board after every single pickup attempt rather than trusting a one-time snapshot, and gives up on
an individual chip (without stalling the rest of the sort) only after it's failed a set number of times. After
sorting finishes, you'll be asked whether to start a game immediately afterward.

### Simulation

**Console-only** (fastest way to exercise `board.py` / `ai.py` / `game_manager.py` logic, no RoboDK needed):
```bash
python main_windows.py --simulate
```

**Full RoboDK simulation** (visual, exercises the real coordinate-transform and motion-sequencing code):
```bash
python -m simulation.robodk_game
```
or, from inside `Game/simulation/`:
```bash
python robodk_game.py
```
Requires a RoboDK station open with the UR3 model, board model, and the reference frames the script expects —
check `simulation/robodk_interface.py` for the exact item names it looks up. `game_manager.py` and `ai.py` run
completely unmodified in this mode; only the camera/arm classes are swapped for RoboDK-backed equivalents.

### Automated tests

```bash
python main_windows.py --test
```
Runs `test_foul_detection.py`'s scripted scenarios (win detection, AI blocking/taking wins, fog-of-war, foul
detection for multi-move and floating-chip cases, and an end-to-end GameManager run).

### Utility / debug scripts

Run these directly (`python <script>.py`), not through `main_windows.py`:

| Script | Purpose |
|---|---|
| `depth_test.py` | Shows the live color feed; click anywhere to print the depth reading at that pixel. Quick sanity check that the depth stream is alive and reasonable. |
| `live_grid_detection.py` | Freeze a live frame, click the board's four corners (TL, TR, BR, BL), then preview the perspective-warped grid classification live — useful for tuning corner detection/HSV thresholds without running a full game. |

## Configuration reference (`config.py`)

| Setting | Meaning |
|---|---|
| `ROBOT_IP` | UR3 controller IP — **double-check this**, see note below |
| `HOME_JOINTS` | Joint configuration the arm returns to between moves |
| `CHIP_ONE_COLOR` / `CHIP_TWO_COLOR` | The two chip color names used throughout (`"red"` / `"yellow"`) |
| `T_CAM_TO_BASE` | 4×4 transform, camera → robot base (from `calibrate_eye_to_hand.py`) |
| `T_BOARD_TO_BASE` | 4×4 transform, board-local → robot base (from `board_calibration.py`) |
| `COLUMN_POSITIONS` | Board-local (x, y, z) entry point for each of the 7 columns |
| `GAME_ORIENTATION` | Fixed tool orientation used for column-insertion moves |
| `TOOL_TCP_OFFSET` | Suction tool's TCP offset relative to the robot flange |
| `SORT_ROI` | Pixel region of interest used for chip detection during sorting |
| `CATCHER_DROP_POSE` / `CATCHER_PICKUP_POSE` | Base-frame poses for the chip-alignment fixture (already measured for the current physical rig) |
| `SIDE_CLEARANCE_OFFSET_M` | Lateral retreat offset applied after a column drop, before returning home |
| `DEFAULT_JOINT_SPEED` / `_ACCEL`, `DEFAULT_LIN_SPEED` / `_ACCEL`, `APPROACH_LIN_SPEED` / `_ACCEL` | Motion tuning for sorting moves |

## Things worth checking / known rough edges

- **`--calibrate` doesn't currently do anything functional** in real mode — `run_real_mode` prints
  "Calibrating column positions..." / "Calibrating pickup positions..." but the actual calls
  (`arm.calibrate_column_positions()`, `arm.calibrate_pickup_positions()`) are commented out, and
  `UR3SuctionArmPublisher` only has stub versions of those methods. Use the dedicated
  `calibration/calibrate_eye_to_hand.py` and `calibration/board_calibration.py` scripts instead — `--calibrate`
  is not a substitute for them.
- **`my_preset.json`** (a RealSense "advanced mode" preset) is present but not applied by default —
  `combined_camera.py`'s `JSON_PRESET_PATH` is `None` unless you pass a path explicitly when constructing
  `CombinedCamera`.
- There is **no physical sensor confirming a successful suction pickup**. Both the game loop and the sorting
  task compensate the same way: after a move, they check whether the board state actually changed, and retry
  with a freshly re-read chip position if not, rather than trusting the arm's motion alone.
- The 2D webcam fallback path (`WebCamCameraInterface`) cannot do real board or chip detection — its
  `read_grid()` and `get_chip_positions()` are explicit placeholders. It exists so the rest of the pipeline can
  still be exercised without a RealSense camera attached, not as a real alternative sensing mode.
