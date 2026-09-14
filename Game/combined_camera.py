"""
combined_camera.py
------------------
Single interface for the RealSense D435 that provides:
  - chip positions (3D) within a sorting ROI (for sorting and game picking)
  - grid state (6x7 matrix) within the blue board ROI (for game state)
The grid state is only updated when `update_grid_state()` or `read_grid()` is called.
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict
from board import PLAYER, ROBOT, EMPTY

# ---------- CHIP DETECTION (same as test_chip_detection.py) ----------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 6
MIN_CONTOUR_AREA = 100
DEPTH_SAMPLE_RADIUS = 3
CAMERA_WARMUP_SECONDS = 10.0   # time to let auto-exposure/white-balance settle
                                # before any mask-based (color) detection runs
JSON_PRESET_PATH = None        # Optional path to a RealSense 'advanced mode'
                                # JSON preset exported from RealSense Viewer
                                # (exposure, gain, laser power, depth accuracy
                                # mode, color correction, etc.). None = skip,
                                # using whatever settings are already active
                                # on the device. Does NOT affect stream
                                # resolution/FPS -- those stay controlled by
                                # CAMERA_WIDTH/HEIGHT/FPS below, independently.

# HSV ranges for chips (tuned)
RED_LOW1 = (0, 80, 65)
RED_HIGH1 = (19, 255, 255)
RED_LOW2 = (170, 50, 50)
RED_HIGH2 = (179, 255, 255)

YELLOW_LOW = (20, 60, 50)
YELLOW_HIGH = (90, 255, 255)

# ---------- GRID DETECTION (same as live_grid_detection.py) ----------
BLUE_LOW_H = 90
BLUE_HIGH_H = 130
BLUE_LOW_S = 50
BLUE_HIGH_S = 255
BLUE_LOW_V = 50
BLUE_HIGH_V = 255

# ---------- CHIP DATA STRUCTURE ----------
@dataclass
class Chip:
    id: int
    color: str
    pixel: Tuple[int, int]
    radius_px: float
    depth_m: float
    position_m: Tuple[float, float, float]   # in camera frame
    distance_m: float

    def to_dict(self):
        d = asdict(self)
        d["pixel"] = list(d["pixel"])
        d["position_m"] = list(d["position_m"])
        return d


# ---------- HELPER FUNCTIONS ----------
def _sample_depth(depth_frame, u, v, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    samples = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            if 0 <= uu < w and 0 <= vv < h:
                d = depth_frame.get_distance(uu, vv)
                if d > 0:
                    samples.append(d)
    return float(np.median(samples)) if samples else 0.0


def _detect_chips_in_mask(mask, color_name, depth_frame, color_intrinsics,
                          next_id, roi=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        # Centroid (mean position of all pixels in the blob), via image
        # moments -- more robust than the enclosing circle's center for
        # anything less than a perfect circle (a chip partially merged
        # with a shadow/reflection in the mask, slightly overlapping a
        # neighbor, or just an imperfect mask edge).
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        u, v = int(round(cx)), int(round(cy))

        # Still compute an enclosing-circle radius purely for
        # Chip.radius_px -- an auxiliary size value, unaffected by the
        # centroid change above.
        _, radius = cv2.minEnclosingCircle(cnt)

        if roi is not None:
            x1, y1, x2, y2 = roi
            if not (x1 <= u <= x2 and y1 <= v <= y2):
                continue

        depth_m = _sample_depth(depth_frame, u, v)
        if depth_m <= 0:
            continue

        X, Y, Z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth_m)
        distance_m = float(np.linalg.norm([X, Y, Z]))

        found.append(Chip(
            id=next_id,
            color=color_name,
            pixel=(u, v),
            radius_px=radius,
            depth_m=depth_m,
            position_m=(X, Y, Z),
            distance_m=distance_m,
        ))
        next_id += 1
    return found, next_id


def _order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_board_contour(img, blue_low, blue_high, min_area=5000):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blue_low, blue_high)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.05 * peri, True)
    if len(approx) != 4:
        hull = cv2.convexHull(largest)
        approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
        if len(approx) != 4:
            return None
    corners = approx.reshape(4, 2)
    corners = _order_corners(corners)
    return corners


def _classify_cell_color(cell_img, chip_ranges):
    """
    Returns the detected chip color as a string ('red', 'yellow'), or
    None if the cell is empty/ambiguous. Does NOT decide which board
    value (PLAYER/ROBOT) this corresponds to -- that depends entirely
    on which color the human actually chose at game start, which this
    function has no way of knowing. See update_grid_state() for where
    that translation actually happens.
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, chip_ranges['red1'][0], chip_ranges['red1'][1])
    mask_red2 = cv2.inRange(hsv, chip_ranges['red2'][0], chip_ranges['red2'][1])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, chip_ranges['yellow'][0], chip_ranges['yellow'][1])
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    total = cell_img.shape[0] * cell_img.shape[1]
    threshold = 0.02 * total
    if red_pixels > threshold and red_pixels > yellow_pixels:
        return 'red'
    elif yellow_pixels > threshold and yellow_pixels > red_pixels:
        return 'yellow'
    else:
        return None


def _apply_json_preset(preset_path):
    """
    Loads a RealSense 'advanced mode' JSON preset (exported from the
    RealSense Viewer) onto the first connected device. Covers
    exposure, gain, laser power, depth accuracy mode, color
    correction, and similar sensor-level settings -- NOT stream
    resolution/FPS, which stay controlled entirely separately, via
    CombinedCamera's own width/height/fps arguments and
    config.enable_stream() calls.

    MUST be called BEFORE pipeline.start(): enabling advanced mode (if
    not already active) causes the device to briefly disconnect and
    reconnect, which would disrupt an already-active stream.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("CombinedCamera: no RealSense device found while "
                            "trying to apply a JSON preset.")
    device = devices[0]

    advnc_mode = rs.rs400_advanced_mode(device)
    if not advnc_mode.is_enabled():
        print("CombinedCamera: advanced mode not enabled -- enabling it now "
              "(device will briefly disconnect/reconnect)...")
        advnc_mode.toggle_advanced_mode(True)
        time.sleep(8)   # give the device time to reconnect before touching it again
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("CombinedCamera: device did not reconnect after "
                                "enabling advanced mode.")
        device = devices[0]
        advnc_mode = rs.rs400_advanced_mode(device)

    with open(preset_path, "r") as f:
        json_text = f.read()

    advnc_mode.load_json(json_text)
    print(f"CombinedCamera: applied JSON preset from '{preset_path}'.")
    # Settle delay AFTER load_json() itself, even when no advanced-mode
    # toggle/reconnect was needed -- applying new sensor settings (laser
    # power, exposure mode, etc.) can leave the sensor briefly unready to
    # stream normally, and starting the pipeline immediately afterward
    # (as the old code did) can produce exactly a "frame didn't arrive"
    # timeout during the very first frame grabs.
    time.sleep(3.0)


# ---------- MAIN COMBINED CAMERA CLASS ----------
class CombinedCamera:
    def __init__(self,
                 chip_roi: Optional[Tuple[int, int, int, int]] = None,
                 transform_matrix: Optional[np.ndarray] = None,
                 blue_hsv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 chip_ranges: Optional[Dict] = None,
                 width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS,
                 warmup_seconds: float = CAMERA_WARMUP_SECONDS,
                 json_preset_path: Optional[str] = JSON_PRESET_PATH,
                 player_color: Optional[str] = None,
                 robot_color: Optional[str] = None):
        self.chip_roi = chip_roi
        self.T_cam_to_base = transform_matrix if transform_matrix is not None else np.eye(4)
        self.blue_low, self.blue_high = blue_hsv if blue_hsv is not None else (
            np.array([BLUE_LOW_H, BLUE_LOW_S, BLUE_LOW_V]),
            np.array([BLUE_HIGH_H, BLUE_HIGH_S, BLUE_HIGH_V])
        )
        # Which board value (PLAYER/ROBOT) each detected color string
        # ("red"/"yellow") maps to -- depends entirely on what the human
        # actually chose at game start, NOT a fixed assumption. If the
        # color choice isn't known yet at construction time (e.g.
        # main_windows.py builds the camera before prompting for color),
        # leave these None and call set_player_robot_colors(...) once
        # it's known, before the first read_grid() call.
        self.player_color = player_color
        self.robot_color = robot_color
        self.chip_ranges = chip_ranges or {
            'red1': (np.array(RED_LOW1), np.array(RED_HIGH1)),
            'red2': (np.array(RED_LOW2), np.array(RED_HIGH2)),
            'yellow': (np.array(YELLOW_LOW), np.array(YELLOW_HIGH)),
        }

        # Apply a JSON preset (if given) BEFORE starting the pipeline --
        # see _apply_json_preset()'s docstring for why the ordering matters.
        if json_preset_path is not None:
            _apply_json_preset(json_preset_path)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)

        sensor = self.profile.get_device().first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 1.0)

        self.align = rs.align(rs.stream.color)

        self.color_intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        # Cache
        self._last_color_image = None
        self._last_chips = []          # transformed chips (base frame)
        self._raw_chips = []           # raw chips (camera frame)
        self._grid_state = None
        self._grid_corners = None
        self._grid_updated = False

        # ---- Let the camera stabilize BEFORE any mask-based (color)
        # detection runs -- auto-exposure and auto-white-balance need real
        # time to settle after the stream starts. Grabbing frames
        # immediately (as _auto_detect_grid_roi() below, and every later
        # update() call, do) can hand the red/yellow/blue HSV thresholds a
        # transient, wrongly-exposed frame, throwing off detection right
        # when it matters most (chip colors, board ROI). Actively pulling
        # and discarding frames for the duration -- rather than a bare
        # time.sleep() -- also prevents the frame queue from backing up
        # with stale buffered frames during the wait, so the next real
        # frame grabbed afterward is actually fresh.
        self._warm_up(warmup_seconds)

        # Auto‑detect grid ROI on first frame
        self._auto_detect_grid_roi()

    def _warm_up(self, seconds):
        """Pulls and discards frames for `seconds` so the camera's
        auto-exposure/auto-white-balance can settle before any real
        detection is attempted on a frame.

        Tolerant of a few transient frame timeouts right at the start --
        this is expected if a JSON preset was just applied (the sensor
        can take a moment to actually begin producing frames after a
        settings change), not necessarily a real failure. Only raises if
        frames still aren't arriving after several retries."""
        if seconds <= 0:
            return
        print(f"CombinedCamera: warming up for {seconds:.0f}s so the image "
              f"can stabilize before detection starts...")
        start = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5
        while time.time() - start < seconds:
            try:
                self.pipeline.wait_for_frames(timeout_ms=10000)
                consecutive_failures = 0
            except RuntimeError as e:
                consecutive_failures += 1
                print(f"CombinedCamera: warm-up frame grab timed out "
                      f"(attempt {consecutive_failures}/{max_consecutive_failures}) "
                      f"-- this can happen right after applying a JSON preset "
                      f"while the sensor settles. Retrying...")
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        "CombinedCamera: camera did not start producing frames "
                        "even after repeated retries during warm-up. If you're "
                        "using json_preset_path, try applying that same preset "
                        "standalone (outside CombinedCamera) first, to confirm "
                        "it doesn't leave the sensor in a bad state on its own."
                    ) from e
                time.sleep(1.0)

    def _auto_detect_grid_roi(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        img = np.asanyarray(color_frame.get_data())
        corners = _find_board_contour(img, self.blue_low, self.blue_high)
        if corners is not None:
            self._grid_corners = corners
        else:
            print("CombinedCamera: Blue board not auto-detected. Set grid corners manually.")

    def set_chip_roi(self, x1, y1, x2, y2):
        self.chip_roi = (x1, y1, x2, y2)

    def set_grid_corners(self, corners):
        self._grid_corners = corners

    def set_player_robot_colors(self, player_color, robot_color):
        """
        Sets which board value each detected color corresponds to for
        grid reading. MUST be called with the human's actual color
        choice before read_grid() is used for a real game -- without
        it, every red-colored cell and every yellow-colored cell can't
        be correctly assigned to PLAYER vs ROBOT, since that assignment
        depends entirely on which color the human picked, not a fixed
        rule.
        """
        self.player_color = player_color
        self.robot_color = robot_color

    def _color_to_board_value(self, detected_color):
        """
        Translates a detected color string ('red'/'yellow'/None) into
        the correct board value (PLAYER/ROBOT/EMPTY), based on
        self.player_color/self.robot_color -- NOT a fixed
        red-is-always-player assumption. If those were never set (e.g.
        set_player_robot_colors() was never called), falls back to
        treating red as PLAYER and yellow as ROBOT with a one-time
        warning -- this fallback exists for backward compatibility, but
        is almost certainly wrong for a real game where the player
        might have chosen yellow.
        """
        if self.player_color is None or self.robot_color is None:
            if not getattr(self, '_warned_missing_colors', False):
                print("CombinedCamera: WARNING -- player_color/robot_color were "
                      "never set via set_player_robot_colors(). Falling back to "
                      "red=PLAYER, yellow=ROBOT, which is almost certainly wrong "
                      "if the player actually chose yellow.")
                self._warned_missing_colors = True
            if detected_color == 'red':
                return PLAYER
            elif detected_color == 'yellow':
                return ROBOT
            else:
                return EMPTY

        if detected_color == self.player_color:
            return PLAYER
        elif detected_color == self.robot_color:
            return ROBOT
        else:
            return EMPTY

    # ---- NEW: update method to grab a fresh frame and detect chips ----
    def update(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        img = np.asanyarray(color_frame.get_data())
        self._last_color_image = img
        depth = depth_frame

        # Chip detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, self.chip_ranges['red1'][0], self.chip_ranges['red1'][1])
        mask_red2 = cv2.inRange(hsv, self.chip_ranges['red2'][0], self.chip_ranges['red2'][1])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, self.chip_ranges['yellow'][0], self.chip_ranges['yellow'][1])

        next_id = 0
        raw_red, next_id = _detect_chips_in_mask(
            mask_red, "red", depth, self.color_intrinsics, next_id, self.chip_roi)
        raw_yellow, next_id = _detect_chips_in_mask(
            mask_yellow, "yellow", depth, self.color_intrinsics, next_id, self.chip_roi)
        self._raw_chips = raw_red + raw_yellow

        # Transform to base frame
        self._last_chips = []
        for chip in self._raw_chips:
            pt = np.array([chip.position_m[0], chip.position_m[1],
                           chip.position_m[2], 1.0])
            pt_base = self.T_cam_to_base @ pt
            transformed = Chip(
                id=chip.id,
                color=chip.color,
                pixel=chip.pixel,
                radius_px=chip.radius_px,
                depth_m=chip.depth_m,
                position_m=(pt_base[0], pt_base[1], pt_base[2]),
                distance_m=chip.distance_m
            )
            self._last_chips.append(transformed)

    def get_chip_positions(self, base_frame=True):
        """
        Returns list of chips in either camera frame (base_frame=False)
        or robot base frame (base_frame=True, default).
        """
        if base_frame:
            return self._last_chips
        else:
            return self._raw_chips

    def update_grid_state(self):
        if self._last_color_image is None or self._grid_corners is None:
            self._grid_state = None
            return

        img = self._last_color_image
        corners = self._grid_corners

        pts_src = np.array(corners, dtype=np.float32)
        out_w, out_h = 700, 600
        pts_dst = np.array([[0,0], [out_w-1,0], [out_w-1,out_h-1], [0,out_h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, M, (out_w, out_h))

        rows, cols = 6, 7
        cell_h = out_h // rows
        cell_w = out_w // cols
        margin = 5
        grid = np.zeros((rows, cols), dtype=np.int32)

        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w + margin
                y1 = r * cell_h + margin
                x2 = (c + 1) * cell_w - margin
                y2 = (r + 1) * cell_h - margin
                if x2 <= x1 or y2 <= y1:
                    continue
                cell_img = warped[y1:y2, x1:x2]
                if cell_img.size == 0:
                    continue
                detected_color = _classify_cell_color(cell_img, self.chip_ranges)
                grid[r, c] = self._color_to_board_value(detected_color)

        # Convert to a plain list of lists -- board.py's create_empty_grid()
        # and everything in game_manager.py works with plain lists
        # throughout (e.g. list != list gives a single bool). Comparing a
        # numpy array against a list with != returns an array of
        # per-element results instead of one bool, which is exactly what
        # crashed _wait_for_camera_change's "if observed != baseline_grid"
        # check.
        self._grid_state = grid.tolist()
        self._grid_updated = True

    def read_grid(self) -> Optional[np.ndarray]:
        # update() captures a fresh frame and sets self._last_color_image,
        # which update_grid_state() depends on. Without calling it here,
        # self._last_color_image stays at its __init__ value of None
        # forever unless SOME caller separately remembers to call
        # camera.update() first -- game_manager.py never did, so
        # read_grid() returned None on every single call, making the
        # entire game loop unable to ever detect a move. Calling it here
        # means read_grid() is correct and self-sufficient regardless of
        # what the caller remembers to do.
        self.update()
        self.update_grid_state()
        return self._grid_state

    def show_live_mask_preview(self, window_name="Chip Mask Preview"):
        """
        Opens a live window showing the camera feed with the current
        red/yellow chip masks overlaid as colored highlights, plus a
        circle and label drawn around each detected chip contour. Uses
        the exact SAME self.chip_ranges thresholds and contour/size
        filtering as update()/get_chip_positions() -- this is a true
        reflection of what real detection is doing, not a separate
        approximation of it, so it's a genuinely trustworthy tool for
        tuning HSV ranges by eye.

        Also draws the chip ROI rectangle and grid corners, if set, for
        full context. This method is self-contained and read-only with
        respect to detection state -- it does not call update() or
        touch self._last_chips/_raw_chips/_grid_state, so it can be run
        at any time without disturbing normal operation.

        Press 'q' or Esc to close the window and return.
        """
        print(f"Live mask preview running in '{window_name}'. Press 'q' or Esc to close.")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask_red1 = cv2.inRange(hsv, self.chip_ranges['red1'][0], self.chip_ranges['red1'][1])
                mask_red2 = cv2.inRange(hsv, self.chip_ranges['red2'][0], self.chip_ranges['red2'][1])
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                mask_yellow = cv2.inRange(hsv, self.chip_ranges['yellow'][0], self.chip_ranges['yellow'][1])

                # Tint matched pixels with colors deliberately DIFFERENT
                # from the chips' own real colors (magenta/cyan rather
                # than red/yellow), so the overlay is obviously an
                # annotation and never confusable with a chip's actual
                # appearance.
                overlay = img.copy()
                overlay[mask_red > 0] = (255, 0, 255)      # magenta over red-matched pixels
                overlay[mask_yellow > 0] = (255, 255, 0)   # cyan over yellow-matched pixels
                blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

                for mask, color_name, draw_color in [(mask_red, "red", (0, 0, 255)),
                                                        (mask_yellow, "yellow", (0, 255, 255))]:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                            continue
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(cx), int(cy))
                        cv2.circle(blended, center, int(radius), draw_color, 2)
                        cv2.putText(blended, color_name, (center[0] - 20, center[1] - int(radius) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1, cv2.LINE_AA)

                if self.chip_roi is not None:
                    x1, y1, x2, y2 = self.chip_roi
                    cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if self._grid_corners is not None:
                    pts = np.array(self._grid_corners, dtype=np.int32)
                    cv2.polylines(blended, [pts], True, (255, 0, 0), 2)

                cv2.imshow(window_name, blended)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:   # 27 = Esc
                    break
        finally:
            cv2.destroyWindow(window_name)

    def close(self):
        self.pipeline.stop()

    def __del__(self):
        self.close()