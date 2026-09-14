"""
calibration.py
Eye-to-hand calibration routine (camera fixed in the workspace, not on
the robot wrist). Move the robot so a calibration marker mounted on the
flange is visible from a range of clearly different poses; this script
collects the robot-pose / observed-marker-pose pairs and solves for
T_CAM_TO_BASE via OpenCV's calibrateHandEye.

Run this once after physically mounting the camera, and again any time
the camera or robot base is moved relative to each other.

Usage:
    1. Attach an ArUco marker (DICT_5X5_50, id doesn't matter) to the
       flange or a known point on the tool.
    2. Measure the marker's side length and set MARKER_LENGTH_M below.
    3. Jog the robot through several distinct poses that keep the marker
       in the camera's view; record each pose's joint values into
       CALIB_POSES_JOINTS.
    4. Run this script. It prints (and saves) the resulting 4x4 transform.
    5. Paste that matrix into config.T_CAM_TO_BASE.
"""

import json
import numpy as np
import cv2

import config
from robot.robot_controller import RobotController
from vision.camera_interface import CameraInterface

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_LENGTH_M = 0.05     # side length of the calibration marker -- MEASURE THIS

# Fill with several taught joint configurations (radians) that keep the
# marker visible from clearly different angles/distances. Aim for at
# least 8-10 poses for a stable solution.
CALIB_POSES_JOINTS = [
    # [j1, j2, j3, j4, j5, j6],
]


def collect_sample(robot, camera, camera_matrix, dist_coeffs):
    color_image, _, _ = camera.get_frames()
    corners, ids, _ = cv2.aruco.detectMarkers(color_image, ARUCO_DICT)
    if ids is None:
        return None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs)
    R_target2cam, _ = cv2.Rodrigues(rvecs[0])
    t_target2cam = tvecs[0].reshape(3, 1)

    flange_pose = robot.current_pose()  # [x, y, z, rx, ry, rz]
    R_gripper2base, _ = cv2.Rodrigues(np.array(flange_pose[3:]))
    t_gripper2base = np.array(flange_pose[:3]).reshape(3, 1)

    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def run_calibration():
    if not CALIB_POSES_JOINTS:
        raise RuntimeError(
            "Fill in CALIB_POSES_JOINTS with several taught joint poses "
            "that keep the calibration marker visible before running this.")

    robot = RobotController()
    camera = CameraInterface()

    intr = camera.intrinsics
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intr.coeffs)

    R_gripper2base_list, t_gripper2base_list = [], []
    R_target2cam_list, t_target2cam_list = [], []

    for joints in CALIB_POSES_JOINTS:
        robot.rtde_c.moveJ(joints, config.DEFAULT_JOINT_SPEED, config.DEFAULT_JOINT_ACCEL)
        sample = collect_sample(robot, camera, camera_matrix, dist_coeffs)
        if sample is None:
            print(f"Marker not visible at pose {joints}, skipping.")
            continue
        Rg, tg, Rt, tt = sample
        R_gripper2base_list.append(Rg)
        t_gripper2base_list.append(tg)
        R_target2cam_list.append(Rt)
        t_target2cam_list.append(tt)

    if len(R_gripper2base_list) < 3:
        raise RuntimeError("Not enough valid samples collected -- add more poses "
                            "or check marker visibility/lighting.")

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI)

    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()

    print("Computed T_CAM_TO_BASE =")
    print(T)

    with open("calibration_result.json", "w") as f:
        json.dump(T.tolist(), f, indent=2)
    print("Saved to calibration_result.json -- paste this matrix into "
          "config.T_CAM_TO_BASE.")

    camera.stop()
    robot.shutdown()


if __name__ == "__main__":
    run_calibration()
