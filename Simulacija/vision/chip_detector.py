"""
vision/chip_detector.py
Finds candidate chip faces in the current color/depth frame using HSV
color segmentation for each configured chip color, filters by shape and
real-world size, and reports each detection's pixel center, 3D position,
radius, and local surface normal -- all in camera frame. sorting_task
converts these into robot base frame using config.T_CAM_TO_BASE.
"""

import cv2
import numpy as np
import config


class Detection:
    def __init__(self, color_name, center_px, radius_px, center_cam, normal_cam, area_px):
        self.color_name = color_name
        self.center_px = center_px          # (x, y) pixel
        self.radius_px = radius_px
        self.center_cam = center_cam        # np.array xyz, camera frame, meters
        self.normal_cam = normal_cam        # np.array xyz unit vector, camera frame
        self.area_px = area_px

    def __repr__(self):
        return f"<Detection {self.color_name} @ {self.center_px} r={self.radius_px:.1f}px>"


def _hsv_mask(hsv_image, hsv_range):
    lo, hi = np.array(hsv_range[0]), np.array(hsv_range[1])
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv_image, lo, hi)
    # Hue wrap-around case (e.g. red spanning ~170-179 and 0-10)
    lo1, hi1 = lo.copy(), hi.copy()
    hi1[0] = 179
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[0] = 0
    return cv2.inRange(hsv_image, lo1, hi1) | cv2.inRange(hsv_image, lo2, hi2)


def _clean_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_chips(color_image, depth_frame, camera):
    """Returns a list of Detection objects, one per chip face found, across
    both configured colors."""
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    detections = []

    color_defs = [
        (config.CHIP_ONE_COLOR, config.CHIP_ONE_HSV_RANGE),
        (config.CHIP_TWO_COLOR, config.CHIP_TWO_HSV_RANGE),
    ]

    for color_name, hsv_range in color_defs:
        mask = _clean_mask(_hsv_mask(hsv, hsv_range))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue  # cheap pixel-space pre-filter; real size checked below

            (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.7:
                continue  # not round enough to be a chip face

            center_cam = camera.pixel_to_point(depth_frame, cx, cy)
            if center_cam is None:
                continue

            normal_cam = camera.local_surface_normal(depth_frame, cx, cy)
            if normal_cam is None:
                continue

            # Convert pixel radius to a metric radius using depth and focal
            # length, then validate against the expected chip diameter range.
            fx = camera.intrinsics.fx
            metric_radius = radius_px * center_cam[2] / fx
            diameter = 2 * metric_radius
            if not (config.CHIP_DIAMETER_RANGE[0] <= diameter <= config.CHIP_DIAMETER_RANGE[1]):
                continue

            detections.append(Detection(color_name, (cx, cy), radius_px,
                                         center_cam, normal_cam, area))

    return detections
