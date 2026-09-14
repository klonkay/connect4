"""
board_calibration.py
Locates the physical game board's coordinate frame in the robot's base
frame, using a single ArUco marker fixed to the board and the camera's
existing hand-eye calibration (config.T_CAM_TO_BASE).

Run this whenever the board is placed or moved. It does NOT move the
robot -- the camera is fixed in the workspace and can see both the board
and the robot's reachable area, so this is a pure vision step.

Usage:
    1. Print an ArUco marker from the config.BOARD_MARKER_DICT dictionary
       with ID config.BOARD_MARKER_ID, sized to config.BOARD_MARKER_LENGTH_M.
    2. Stick it flat to the board at a known, fixed spot, with the
       marker's printed edges aligned to however you want the board's
       local X/Y axes to run (this defines the frame that the
       BOARD_ROBOT_HALF_CORNER_A/B and BOARD_PLAYER_HALF_CORNER_A/B
       values in config.py are measured relative to).
    3. Make sure config.T_CAM_TO_BASE is already filled in from
       calibration.py -- this script builds on top of it.
    4. Run this script with a clear view of the marker (no chips or
       hands covering it).

    python board_calibration.py

Then paste the printed matrix into config.T_BOARD_TO_BASE.
"""

import json
import numpy as np
import cv2

import config
from vision.camera_interface import CameraInterface

ARUCO_DICT_MAP = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def find_board_pose_in_camera(camera, num_samples=15):
    """Detects the board marker over several frames and averages the pose
    for stability, returning a 4x4 transform: board frame -> camera frame."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[config.BOARD_MARKER_DICT])
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    intr = camera.intrinsics
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intr.coeffs)

    rvecs_collected, tvecs_collected = [], []

    for _ in range(num_samples):
        color_image, _, _ = camera.get_frames()
        if color_image is None:
            continue
        corners, ids, _ = detector.detectMarkers(color_image)
        if ids is None:
            continue
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == config.BOARD_MARKER_ID:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], config.BOARD_MARKER_LENGTH_M,
                    camera_matrix, dist_coeffs)
                rvecs_collected.append(rvec[0][0])
                tvecs_collected.append(tvec[0][0])

    if not rvecs_collected:
        raise RuntimeError(
            f"Marker ID {config.BOARD_MARKER_ID} was never detected. Check "
            f"that it's in view, unobstructed, well lit, and that "
            f"BOARD_MARKER_DICT / BOARD_MARKER_ID match the marker you printed.")

    # Average translation directly; average rotation via its rotation
    # matrices to avoid axis-angle wraparound issues.
    t_avg = np.mean(tvecs_collected, axis=0)
    R_sum = np.zeros((3, 3))
    for rvec in rvecs_collected:
        R, _ = cv2.Rodrigues(rvec)
        R_sum += R
    # Re-orthonormalize the averaged rotation matrix via SVD.
    u, _, vt = np.linalg.svd(R_sum)
    R_avg = u @ vt

    T_board_to_cam = np.eye(4)
    T_board_to_cam[:3, :3] = R_avg
    T_board_to_cam[:3, 3] = t_avg
    return T_board_to_cam, len(rvecs_collected)


def run_board_calibration():
    if np.allclose(config.T_CAM_TO_BASE, np.eye(4)):
        print("Warning: config.T_CAM_TO_BASE is still the identity placeholder. "
              "Run calibration.py first, or this result will be meaningless.")

    camera = CameraInterface()
    try:
        T_board_to_cam, n = find_board_pose_in_camera(camera)
        print(f"Averaged pose over {n} valid marker detections.")

        T_board_to_base = config.T_CAM_TO_BASE @ T_board_to_cam

        print("Computed T_BOARD_TO_BASE =")
        print(T_board_to_base)

        with open("board_calibration_result.json", "w") as f:
            json.dump(T_board_to_base.tolist(), f, indent=2)
        print("Saved to board_calibration_result.json -- paste this matrix "
              "into config.T_BOARD_TO_BASE.")
    finally:
        camera.stop()


if __name__ == "__main__":
    run_board_calibration()
