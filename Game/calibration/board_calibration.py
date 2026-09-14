"""
board_calibration.py
--------------------
Locates the board's ArUco marker and computes T_BOARD_TO_BASE.
Requires config.T_CAM_TO_BASE to be already set (from calibration.py).
Uses RealSense pipeline directly (no camera class dependency).
"""

import json
import numpy as np
import cv2
import pyrealsense2 as rs
import time
from config import T_CAM_TO_BASE

# ---------- USER SETTINGS ----------
# Must match the marker printed and stuck to the board.
BOARD_MARKER_DICT = cv2.aruco.DICT_5X5_50
BOARD_MARKER_ID = 0
BOARD_MARKER_LENGTH = 0.0995          # side length in meters – MEASURE YOUR MARKER
NUM_SAMPLES = 15                    # number of frames to average over
MAX_REPROJ_ERROR_PX = 1.5           # frames worse than this are discarded before averaging
WIDTH, HEIGHT, FPS = 1280, 720, 6   # RealSense color stream settings
TIME_DELAY = 0.5                    # seconds to wait after each frame capture (for camera stabilization)

# ---------- Helper function ----------
def detect_board_marker(pipeline, align, intrinsics, marker_id, marker_length, num_samples,
                         max_reproj_error_px=MAX_REPROJ_ERROR_PX):
    """
    Detect the board marker over several frames, average the pose,
    and return T_board_to_cam (4x4).

    Uses subpixel corner refinement and IPPE_SQUARE -- both meaningfully
    improve accuracy for a single planar square marker, especially at an
    angle, where corner localization (fewer pixels per foreshortened
    edge) is the main source of error. Also computes a reprojection
    error for every sample so you have an objective number for how
    trustworthy each read actually was, instead of just trusting the
    preview image -- frames above max_reproj_error_px are discarded
    before averaging rather than silently pulling the average off.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(BOARD_MARKER_DICT)
    params = cv2.aruco.DetectorParameters()
    # Subpixel corner refinement -- the single biggest lever for
    # improving accuracy at an oblique angle, where the marker's edges
    # are foreshortened and the default corner finder has less to work
    # with.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.array(intrinsics.coeffs, dtype=np.float32)

    half = marker_length / 2.0
    # Corner order MUST match cv2.aruco's own detected-corner order
    # exactly: top-left -> top-right -> bottom-right -> bottom-left, in
    # image space. Confirmed via real hardware: the previous ordering
    # here traced the square in the OPPOSITE winding direction, and
    # produced a consistent ~245-250px reprojection error across every
    # single captured frame (near-zero is expected for a correct read) --
    # exactly the signature of a correspondence mismatch, not detection
    # noise. Fixed to match calibrate_eye_to_hand.py's confirmed-correct
    # ordering.
    object_pts = np.array([[-half,  half, 0],   # top-left
                           [ half,  half, 0],   # top-right
                           [ half, -half, 0],   # bottom-right
                           [-half, -half, 0]], dtype=np.float32)  # bottom-left

    rvecs = []
    tvecs = []
    reproj_errors = []

    print(f"Detecting board marker ID {marker_id}... (averaging over {num_samples} frames)")
    for frame_idx in range(num_samples):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            print(f"  Frame {frame_idx+1}: no color frame, skipping.")
            continue
        color_image = np.asanyarray(color_frame.get_data())

        corners, ids, _ = detector.detectMarkers(color_image)
        if ids is None:
            print(f"  Frame {frame_idx+1}: no markers detected.")
            continue

        time.sleep(TIME_DELAY)  # allow camera to stabilize before next frame

        for i, mid in enumerate(ids.flatten()):
            if mid == marker_id:
                marker_corners = corners[i][0]   # shape (4,2)
                success, rvec, tvec = cv2.solvePnP(
                    object_pts, marker_corners, K, dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not success:
                    print(f"  Frame {frame_idx+1}: solvePnP failed, skipping.")
                    break

                # Reprojection error: project object_pts back through the
                # estimated pose and compare to where the corners were
                # actually detected. A clean, trustworthy read should be
                # well under 1px; an overly oblique or shaky detection
                # will show it here as a number, rather than you having
                # to guess from the preview image alone.
                projected, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
                projected = projected.reshape(-1, 2)
                err = float(np.linalg.norm(projected - marker_corners, axis=1).mean())

                if err > max_reproj_error_px:
                    print(f"  Frame {frame_idx+1}: detected, but reprojection error "
                          f"{err:.2f}px exceeds {max_reproj_error_px}px -- discarding.")
                    break

                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
                reproj_errors.append(err)
                print(f"  Frame {frame_idx+1}: marker detected, reprojection error {err:.2f}px.")

                # Show the detected marker AND its estimated pose axes for
                # visual confirmation -- this is also the concrete way to
                # sanity-check the corner ordering above: the axes should
                # sit flush on the marker's visible edges, not skewed.
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.drawFrameAxes(color_image, K, dist, rvec, tvec, marker_length * 0.5)
                cv2.imshow("Board Detection", color_image)
                cv2.waitKey(1)
                break

    cv2.destroyAllWindows()

    if not rvecs:
        raise RuntimeError(
            f"Board marker ID {marker_id} not detected clearly enough in any frame "
            f"(either not visible, or every read exceeded the "
            f"{max_reproj_error_px}px reprojection error threshold). "
            f"Check lighting, angle, and visibility."
        )

    print(f"\nKept {len(rvecs)}/{num_samples} frames "
          f"(mean reprojection error: {np.mean(reproj_errors):.2f}px, "
          f"max: {np.max(reproj_errors):.2f}px)")

    # Average translation
    t_avg = np.mean(tvecs, axis=0)
    # Average rotation via SVD of summed rotation matrices
    R_sum = np.zeros((3, 3))
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        R_sum += R
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt

    T_board_to_cam = np.eye(4)
    T_board_to_cam[:3, :3] = R_avg
    T_board_to_cam[:3, 3] = t_avg
    return T_board_to_cam

# ---------- Main ----------
def main():
    # Check if T_CAM_TO_BASE is identity (placeholder)
    if np.allclose(T_CAM_TO_BASE, np.eye(4)):
        print("WARNING: config.T_CAM_TO_BASE is still the identity matrix.")
        print("Run calibration.py first to compute the true transform.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Aborting.")
            return

    # Start RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    profile = pipeline.get_active_profile()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
        T_board_to_cam = detect_board_marker(
            pipeline, align, intrinsics,
            BOARD_MARKER_ID, BOARD_MARKER_LENGTH, NUM_SAMPLES,
            max_reproj_error_px=MAX_REPROJ_ERROR_PX
        )
        print("Board marker detected.")
    finally:
        pipeline.stop()

    # Transform to base frame using the known camera-to-base transform
    T_board_to_base = T_CAM_TO_BASE @ T_board_to_cam

    print("\nComputed T_BOARD_TO_BASE =\n", T_board_to_base)
    with open("board_calibration_result.json", "w") as f:
        json.dump(T_board_to_base.tolist(), f, indent=2)
    print("Saved to board_calibration_result.json.")
    print("Copy this matrix into config.T_BOARD_TO_BASE.")

if __name__ == "__main__":
    main()
