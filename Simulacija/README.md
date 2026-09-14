# UR3 Chip Sorting System

Resets a game board to its correct starting layout using a UR3 arm, a 3D
(RGB-D) camera, and a suction end effector. Each color has a fixed half
of the board it belongs on; at the start of a game the player picks
which color they're using, and the robot moves every chip that's
currently on the wrong half (either color, wherever it happens to be
after a messy reset) to its correct side, leaving already-correctly-
placed chips untouched. Each chip has one larger flat monochromatic
circular face and one smaller monochromatic face with an off-color
border; the system suctions whichever face is currently up and reads its
main color to identify it.

## Project layout

```
chip_sorting_project/
├── config.py                  # ALL tunable values live here (see below)
├── calibration.py              # one-time camera-to-robot hand-eye calibration
├── board_calibration.py        # one-time (or after any board move) board-pose calibration
├── main.py                     # entry point -- prompts for the player's color
├── requirements.txt
├── robot/
│   └── robot_controller.py     # ur_rtde wrapper: motion + suction I/O
├── vision/
│   ├── camera_interface.py     # RealSense wrapper: frames, 3D points, normals
│   ├── chip_detector.py        # HSV segmentation + shape/size filtering
│   └── color_classifier.py     # color confidence + border-ring detection
├── planning/
│   ├── geometry_utils.py       # normal -> pose, frame transforms
│   ├── path_planner.py         # pick/place waypoint sequences, board-clearance-aware travel height
│   └── placement_manager.py    # fixed robot/player half rectangles, slot allocation
├── task/
│   └── sorting_task.py         # the detect->find-misplaced->pick->place->repeat state machine
└── simulation/
    ├── robodk_interface.py     # RoboDK bridge: motion + ground-truth chip/board poses + collision checks
    ├── sim_task.py              # same game logic, run against a RoboDK station
    └── diagnose_alignment.py   # standalone diagnostic: TCP/frame/pose sanity checks + live visual test
```

## How the pipeline works, end to end

1. **Choose a color**: `main.py` (or `simulation/sim_task.py` in
   simulation) prompts for which color the player is using for this
   game. This fixes which half each color belongs on: the player's color
   belongs on the player's half, the other color belongs on the robot's
   half.
2. **Detect**: `chip_detector.detect_chips` segments the current color
   frame by HSV, keeps only sufficiently circular, correctly-sized
   contours, and looks up each one's 3D position and local surface
   normal from the depth data -- for BOTH colors, since a messy reset
   can leave either color's chips on either half.
3. **Find what's misplaced**: `sorting_task._pending_targets` checks
   each detected chip's current half (via
   `placement_manager.half_of_point`) against its correct half for the
   chosen color. Chips already on their correct half are skipped
   entirely -- never selected, never touched, counted as already sorted.
4. **Select target**: picks the closest-to-camera chip among the
   pending (misplaced) ones.
5. **Plan pick path**: `path_planner.build_pick_path` builds a
   travel -> approach -> contact -> retreat sequence from the chip's
   position/normal and the `TOOL_*` dimensions in `config.py`. The
   travel height also clears `config.BOARD_OBSTRUCTION_HEIGHT`, not just
   the chip's own local clearance -- see "Board collision avoidance" below.
6. **Suction**: the robot enables suction before contact, seats the cup,
   and (if a vacuum sensor is wired) confirms the seal. A failed seal
   triggers a retry.
7. **Plan release path**: `placement_manager.next_slot` returns the next
   free grid slot in the chip's correct half, starting from that half's
   center and spiraling outward. `path_planner.build_place_path` turns
   that into a travel -> approach -> release -> retreat sequence.
8. **Release**: suction switches off just above full contact.
9. **Check completion**: after each chip, the system re-detects; once no
   chip of either color is on the wrong half, it returns to
   `HOME_JOINTS`.

## Board collision avoidance

Two layers, since neither alone is sufficient:

- **`path_planner._effective_travel_height()`** ensures every lateral
  ("travel") leg clears whichever is larger of the tool's usual safe
  travel height, or `config.BOARD_OBSTRUCTION_HEIGHT` (your board's
  tallest raised feature) plus margin. This is a simple "fly high enough
  over the whole board" strategy -- not full 3D obstacle-aware planning,
  just a uniform ceiling. It assumes no part of the board within reach
  exceeds that height anywhere.
- **In simulation**, `RoboDKInterface` additionally enables RoboDK's own
  collision engine (`setCollisionActive`) and checks for collisions
  after every move, printing a warning if one's detected -- this catches
  the arm actually hitting your imported board assembly's real geometry,
  which our own path planning doesn't reason about directly. Make sure
  your board object is included in RoboDK's Collision Map (Tools >
  Collision Map). Note: `setCollisionActive`/`Collisions()` are RoboDK's
  standard collision-checking calls, but if your installed RoboDK
  version has renamed or restructured them, this will error at startup
  rather than silently doing nothing -- if that happens, check RoboDK's
  own API docs/autocomplete for the current equivalents in your version.
- **On the real robot**, there's no equivalent automatic check yet --
  `BOARD_OBSTRUCTION_HEIGHT` needs to be a genuinely conservative measure
  of your physical board, and ArUco-based board calibration
  (`board_calibration.py`) is what keeps the robot's understanding of
  where the board actually is up to date if it moves.

## Setup order

1. **Robot network**: set `config.ROBOT_IP`, teach a safe `HOME_JOINTS`
   pose clear of the camera and board.
2. **Suction tool dimensions**: fill in `TOOL_FLANGE_TO_CUP_TIP`,
   `TOOL_CUP_RADIUS`, and `TOOL_CUP_COMPRESSION` once measured. Every
   path and TCP registration derives from these automatically.
3. **Suction I/O**: in `robot/robot_controller.py`, set
   `SUCTION_DO_INDEX` and (once wired) `VACUUM_SENSOR_DI_INDEX`.
4. **Camera calibration**: run `calibration.py` and paste the resulting
   matrix into `config.T_CAM_TO_BASE`.
5. **Chip colors**: `CHIP_ONE_COLOR` / `CHIP_TWO_COLOR` and their HSV
   ranges in `config.py` should match your real chips under your actual
   lighting -- tune with a throwaway script that prints HSV under the
   mouse cursor on a live feed.
6. **Board calibration**: stick a printed ArUco marker to the physical
   board at a fixed spot, set `BOARD_MARKER_DICT` / `BOARD_MARKER_ID` /
   `BOARD_MARKER_LENGTH_M` in `config.py`, and measure
   `BOARD_ROBOT_HALF_CORNER_A` / `_B` and `BOARD_PLAYER_HALF_CORNER_A` /
   `_B` **on the board itself** -- these are two independent rectangles,
   one per half, and don't need to share an exact common edge. Also set
   `SURFACE_Z_BOARD` and `BOARD_OBSTRUCTION_HEIGHT`. Then run
   `board_calibration.py` and paste the result into
   `config.T_BOARD_TO_BASE`. If the board is ever bumped or moved,
   re-run `board_calibration.py` only.
7. Run `python main.py` and enter the player's color when prompted.

## Simulating in RoboDK (before touching the real robot)

`simulation/` lets you validate the actual pick/place path-planning logic
against your UR3 model in RoboDK, without a real robot or camera. It
reuses `planning/path_planner.py` and `planning/placement_manager.py`
completely unchanged, and swaps only motion + perception:

| Real pipeline | Simulated pipeline |
|---|---|
| `robot/robot_controller.py` (ur_rtde) | `simulation/robodk_interface.py` (RoboDK API) |
| `vision/camera_interface.py` + `chip_detector.py` | `RoboDKInterface.list_chip_items()` — reads ground-truth chip poses/colors straight from the station |
| `task/sorting_task.py` | `simulation/sim_task.py` |

This validates your **motion, sorting, and collision-avoidance logic** —
waypoint heights, tool-offset math, retry flow, fixed-half placement —
but does not exercise the real HSV/depth vision pipeline, since it reads
chip poses directly from the station instead of a simulated camera feed.

### Station setup

1. Install RoboDK and the `robodk` Python package: `pip install robodk`.
2. Add your robot from the RoboDK library, and name the item to match
   `ROBODK_ROBOT_NAME` in `simulation/robodk_interface.py` (defaults to
   `"UR3"`).
3. Add your suction tool model, attach it to the robot's flange, and
   name it to match `ROBODK_TOOL_NAME` (`"SuctionTool"`). Its own TCP
   setting inside RoboDK doesn't need manual configuration —
   `config.TOOL_TCP_OFFSET` is pushed in automatically as the active TCP.
4. Import your full-size board assembly. Make sure it's included in
   RoboDK's Collision Map (Tools > Collision Map) so the built-in
   collision check (see "Board collision avoidance" above) actually
   catches the arm hitting it.
5. Scatter copies of your chip model across the board — for a realistic
   reset test, mix both colors across both halves, including some
   already correctly placed. Name each one `chip_<color>_<n>` — e.g.
   `chip_red_1`, `chip_yellow_1` — using the same color strings as
   `config.CHIP_ONE_COLOR` / `CHIP_TWO_COLOR`. Make sure each chip's
   local **+Z axis points out of its flat face** — that's what stands in
   for a camera-measured surface normal.
6. Add a Frame item named `"BoardFrame"` on the board (a corner is
   easiest to measure from), oriented however you want the board's local
   X/Y to run. `simulation/robodk_interface.py` reads this frame's
   ground-truth pose automatically and uses it in place of
   `config.T_BOARD_TO_BASE` — no calibration script needed in sim.
7. Optionally add a `Target` item named `"Home"` for the resting pose;
   otherwise the sim falls back to `config.HOME_JOINTS`.
8. Double-check `config.py`'s `BOARD_ROBOT_HALF_CORNER_A/B`,
   `BOARD_PLAYER_HALF_CORNER_A/B`, `SURFACE_Z_BOARD`, and
   `BOARD_OBSTRUCTION_HEIGHT` match your actual board, and
   `PLACEMENT_GRID_PITCH` is larger than your chip model's diameter.
9. With the station open in RoboDK, run:
   ```bash
   python -m simulation.sim_task
   ```
   It will prompt you for which color the player is using before starting.

Nothing in the algorithm code needs to change for this — the same
`path_planner.py` / `placement_manager.py` used for the real robot run
unmodified; only the station setup above (chip poses, board geometry,
`BoardFrame`) determines what they compute.

### A unit quirk worth knowing

RoboDK's `robomath.UR_2_Pose()` / `Pose_2_UR()` helpers treat the
translation part as **millimeters**, even though they're named after and
shaped like UR's own pose format (which real UR controllers report in
**meters**). This is a known mismatch in RoboDK's API, not a typo on our
end. `robodk_interface.py` already handles the conversion — if you ever
extend it, keep converting meters → millimeters before calling
`UR_2_Pose()`.

### If you want to test the real vision pipeline too

RoboDK's built-in 2D camera simulation (Tools > Camera) can render a
color image of the scene, which you could feed into the real
`vision/chip_detector.py` HSV pipeline for a closer end-to-end test — but
it won't produce real depth data, so you'd need to fake the depth/normal
lookup (e.g. using each chip's known Z height and orientation in the
station) rather than relying on `camera_interface.local_surface_normal`.
This is a reasonable next step once the ground-truth motion test above
is working cleanly, but isn't required to validate path planning.

## Notes / things to revisit

- `MAX_SORT_RETRIES`, `PLACEMENT_GRID_PITCH`, and detection thresholds in
  `config.py` are reasonable starting points — tune them once you're
  running against real chips and real lighting.
- The retry logic in `sorting_task.py` re-runs detection between
  attempts in case a failed pick nudges the chip.
- If you add more than two chip colors later, extend the color-def list
  in `chip_detector.py` and decide how a "move this color" instruction
  should behave with more than two halves to flip between (the current
  split logic is built for exactly two).
