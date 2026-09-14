"""
Eye-to-Hand Calibration: Fixed RealSense Camera -> UR3 (CB3) Base Frame
==========================================================================

Solves for the static transform T_cam_to_base between a FIXED RealSense
camera and the robot base, using an ArUco marker rigidly attached to the
robot's end-effector (flange, gripper, or any rigid tool mount).

Method: classic AX = XB hand-eye calibration, using the eye-to-hand trick
of inverting gripper->base poses to base->gripper before calling
cv2.calibrateHandEye(), which then returns cam->base directly instead of
cam->gripper.

Uses the PARK method rather than TSAI: independent testing (comparisons
across all five cv2.calibrateHandEye methods, and a still-open OpenCV
GitHub issue specifically about TSAI producing wrong values in eye-to-hand
setups) found TSAI consistently less reliable for this exact configuration.

Also captures the robot's TCP pose at the SAME INSTANT as the camera
frame, rather than after the preview window and keep/discard decision --
see the CAPTURE LOOP section below for why this matters for precision.

Uses subpixel corner refinement: this is what actually lets a higher
COLOR_WIDTH/COLOR_HEIGHT translate into genuinely better corner
localization, rather than just more raw pixels processed the same
imprecise way. Also computes and displays a reprojection error (pixels)
for every capture, alongside marker distance, so keep/discard decisions
are backed by an objective number rather than the preview image alone --
this matters more at higher resolution, where a real detection problem
is less likely to be visually obvious just by looking.

Dependencies:
    pip install opencv-contrib-python numpy pyrealsense2 urx

Usage:
    1. Edit the CONFIGURATION block below.
    2. Print an ArUco marker (dictionary + ID must match config) and
       mount it rigidly on the robot end-effector.
    3. Run this script: python calibrate_eye_to_hand.py
    4. Put the robot in freedrive/teach mode on the pendant.
    5. Move the robot to a new pose, press ENTER to capture. Repeat
       15-25 times with varied positions AND orientations, keeping the
       marker visible to the camera each time.
    6. For each capture, the annotated image will be shown - press any
       key in the image window to close it, then decide in the terminal.
    7. Press 'q' + ENTER when done. The script solves the calibration
       and writes the result to OUTPUT_JSON.
"""

import time
import json
import datetime

import numpy as np
import cv2
import pyrealsense2 as rs

# --- Compatibility shim ---
import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import urx

# ============================== CONFIGURATION ==============================
ROBOT_IP = "192.168.40.27"          # IP address of the UR3 CB3 controller
MARKER_ID = 1                          # ArUco marker ID to track
MARKER_LENGTH_M = 0.0495            # Marker side length, in meters -- measured
                                     # with calipers.
ARUCO_DICT = cv2.aruco.DICT_5X5_50  # Must match the marker you printed
MIN_POSES = 15                      # Minimum number of poses before solving
CALIB_METHOD = cv2.CALIB_HAND_EYE_PARK  # PARK, not TSAI -- see module docstring
OUTPUT_JSON = "cam_to_base_transform.json"
COLOR_WIDTH, COLOR_HEIGHT, FPS = 1280, 720, 6
# =============================================================================


def list_supported_color_profiles(device):
    """Return every (width, height, fps, format) color stream profile the device supports."""
    profiles = set()
    for sensor in device.query_sensors():
        for sp in sensor.get_stream_profiles():
            if sp.stream_type() == rs.stream.color:
                vp = sp.as_video_stream_profile()
                profiles.add((vp.width(), vp.height(), vp.fps(), vp.format()))
    return sorted(profiles, key=lambda p: (p[0], p[1], p[2], int(p[3])))


def init_realsense():
    """Start the RealSense color stream and return camera intrinsics."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense device detected. Check that the camera is plugged in "
            "(preferably a USB3 port), that no other program (e.g. RealSense "
            "Viewer) is holding it open, and that it shows up in the RealSense "
            "Viewer app before running this script."
        )

    device = devices[0]
    print(f"Found RealSense device: {device.get_info(rs.camera_info.name)} "
          f"(serial {device.get_info(rs.camera_info.serial_number)})")

    supported = list_supported_color_profiles(device)
    requested = (COLOR_WIDTH, COLOR_HEIGHT, FPS, rs.format.bgr8)

    pipeline = rs.pipeline()
    config = rs.config()

    if requested in supported:
        width, height, fps, fmt = requested
    else:
        print(f"WARNING: {COLOR_WIDTH}x{COLOR_HEIGHT} @ {FPS}fps (bgr8) is not "
              f"supported by this camera. Falling back to a supported profile.")
        bgr8_profiles = [p for p in supported if p[3] == rs.format.bgr8]
        if not bgr8_profiles:
            raise RuntimeError(
                "No bgr8 color profiles available on this device. "
                f"Supported color profiles: {supported}"
            )
        width, height, fps, fmt = bgr8_profiles[-1]
        print(f"Using fallback profile: {width}x{height} @ {fps}fps")

    config.enable_stream(rs.stream.color, width, height, fmt, fps)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1]
    ], dtype=np.float64)

    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)

    return pipeline, camera_matrix, dist_coeffs


def get_marker_pose(color_image, camera_matrix, dist_coeffs, marker_length, marker_id, aruco_dict_type):
    """Detect the given ArUco marker and return (rvec, tvec, reprojection_error_px,
    annotated_image) or (None, None, None, None)."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    # Subpixel corner refinement -- this is what actually lets a higher
    # COLOR_WIDTH/COLOR_HEIGHT translate into better corner localization,
    # rather than just more raw pixels processed the same imprecise way.
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None, None, None

    ids_flat = ids.flatten().tolist()
    if marker_id not in ids_flat:
        return None, None, None, None

    idx = ids_flat.index(marker_id)
    marker_corners = corners[idx]

    half = marker_length / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, marker_corners[0], camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        return None, None, None, None

    # Reprojection error: project obj_points back through the estimated
    # pose and compare to where the corners were actually detected. A
    # clean, trustworthy read should be well under 1px -- shown alongside
    # marker distance below so keep/discard decisions have an objective
    # number to go on, not just the preview image.
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_err = float(np.linalg.norm(projected.reshape(-1, 2) - marker_corners[0], axis=1).mean())

    annotated = color_image.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

    return rvec, tvec, reproj_err, annotated


def get_robot_pose(robot):
    """Return (R, t) for the robot's current TCP pose in the base frame."""
    pose = robot.getl()  # [x, y, z, rx, ry, rz], meters + axis-angle radians
    t = np.array(pose[:3]).reshape(3, 1)
    rvec = np.array(pose[3:6]).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R, t


def show_preview(annotated_image, display_time=4.0):
    """Display the annotated image for `display_time` seconds, then close it."""
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 1200, 675)
    cv2.imshow("Preview", annotated_image)
    # Let the window update and show the image
    cv2.waitKey(1)
    # Wait for the specified time, then close
    time.sleep(display_time)
    cv2.destroyWindow("Preview")


def main():
    print(f"Connecting to robot at {ROBOT_IP} ...")
    robot = urx.Robot(ROBOT_IP)
    time.sleep(0.5)

    print("Starting RealSense pipeline ...")
    pipeline, camera_matrix, dist_coeffs = init_realsense()
    time.sleep(1.0)  # let auto-exposure settle

    R_gripper2base_list, t_gripper2base_list = [], []
    R_target2cam_list, t_target2cam_list = [], []

    print("\n=== Eye-to-Hand Calibration ===")
    print(f"Marker ID: {MARKER_ID}, size: {MARKER_LENGTH_M * 100:.2f} cm")
    print(f"Need at least {MIN_POSES} captured poses.\n")
    print("Instructions:")
    print(" 1. Put the robot in FREEDRIVE / teach mode on the pendant.")
    print(" 2. Move the robot so the marker is clearly visible to the camera.")
    print(" 3. Press ENTER here to capture that pose.")
    print(" 4. A preview window will appear. Press any key to close it, then decide.")
    print(" 5. Type 'k' to keep, 'd' to discard, or 'q' to quit early.\n")

    sample_index = 0
    quit_early = False

    robot.set_freedrive(1)

    try:
        while not quit_early:
            cmd = input(f"[{len(R_gripper2base_list)} kept] ENTER to capture, 'q' to finish: ")
            if cmd.strip().lower() == "q":
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("  No color frame received, try again.")
                continue

            # ---- CAPTURE LOOP: read the robot pose at THE SAME INSTANT
            # as the camera frame -- before marker detection, the preview
            # window, or waiting on your keep/discard decision have any
            # chance to let the arm drift (it's in freedrive, so it's
            # easy to nudge or for it to settle slightly). Even a small
            # mismatch between "where the robot was when the photo was
            # taken" and "where the robot pose was recorded" directly
            # corrupts a sample, since the whole method depends on both
            # readings describing the exact same physical configuration.
            # This candidate is only kept if you decide to keep the
            # sample below -- nothing is committed yet.
            R_gripper2base_candidate, t_gripper2base_candidate = get_robot_pose(robot)

            color_image = np.asanyarray(color_frame.get_data())

            rvec, tvec, reproj_err, annotated = get_marker_pose(
                color_image, camera_matrix, dist_coeffs,
                MARKER_LENGTH_M, MARKER_ID, ARUCO_DICT
            )

            if rvec is None:
                print("  Marker not detected — adjust pose and try again.")
                continue

            sample_index += 1
            marker_distance = np.linalg.norm(tvec)

            # Show the preview for 2 seconds, then close
            show_preview(annotated, display_time=2.0)

            # Now ask for decision in the terminal -- reprojection error
            # shown alongside distance so the decision isn't based on the
            # preview image alone.
            decision = input(
                f"  Sample {sample_index}: distance {marker_distance:.3f} m, "
                f"reprojection error {reproj_err:.2f}px "
                f"({'looks good' if reproj_err < 1.0 else 'check this one closely'}). "
                f"Keep? (k/d/q): "
            ).strip().lower()
            robot.set_freedrive(1)
            if decision == 'q':
                quit_early = True
                break
            elif decision == 'd':
                print("  Sample discarded.")
                continue
            elif decision != 'k':
                print("  Invalid input, treating as discard.")
                continue
            # else keep

            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec

            R_gripper2base_list.append(R_gripper2base_candidate)
            t_gripper2base_list.append(t_gripper2base_candidate)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(t_target2cam)

            fname = f"capture_kept_{len(R_gripper2base_list):02d}.png"
            cv2.imwrite(fname, annotated)
            print(f"  Kept sample {len(R_gripper2base_list)} -> saved {fname}")

    finally:
        robot.set_freedrive(0)
        pipeline.stop()
        robot.close()
        cv2.destroyAllWindows()

    n = len(R_gripper2base_list)
    if n < MIN_POSES:
        print(f"\nOnly {n} poses kept, need at least {MIN_POSES}. Aborting without solving.")
        return

    print(f"\nCollected {n} good poses. Solving hand-eye calibration ...")

    # --- Eye-to-hand trick ---
    R_base2gripper_list, t_base2gripper_list = [], []
    for R, t in zip(R_gripper2base_list, t_gripper2base_list):
        R_inv = R.T
        t_inv = -R_inv @ t
        R_base2gripper_list.append(R_inv)
        t_base2gripper_list.append(t_inv)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper_list, t_base2gripper_list,
        R_target2cam_list, t_target2cam_list,
        method=CALIB_METHOD
    )

    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()

    print("\nResult T_cam_to_base:")
    print(T_cam2base)

    result = {
        "description": "Fixed RealSense camera to UR3 base transform (eye-to-hand)",
        "computed_at": datetime.datetime.now().isoformat(),
        "num_poses_used": n,
        "method": "PARK",
        "marker_id": MARKER_ID,
        "marker_length_m": MARKER_LENGTH_M,
        "T_cam_to_base": T_cam2base.tolist(),
        "R_cam_to_base": R_cam2base.tolist(),
        "t_cam_to_base": t_cam2base.flatten().tolist(),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved calibration to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()