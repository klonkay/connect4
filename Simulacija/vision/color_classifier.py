"""
vision/color_classifier.py
Secondary color verification and face-identification helpers. The main
color decision already happens in chip_detector.py via HSV masks; this
module adds:
  - a confidence score for how "solid" a detected region's color match is
  - detection of the off-color border ring, which tells you whether the
    currently-visible face is the smaller bordered face or the larger
    plain face (informational/logging -- suction works on either).
"""

import cv2
import numpy as np
import config


def dominant_color_fraction(mask):
    """Fraction of a region's bounding box that actually matched the HSV
    range -- a simple confidence score for how solid/clean the read is."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    return len(xs) / max(bbox_area, 1)


def has_border_ring(hsv_image, cnt):
    """
    Checks a thin ring just outside a detected main-color contour for
    pixels matching config.CHIP_BORDER_HSV_RANGE. If present, the visible
    face is likely the smaller bordered face rather than the larger plain
    face -- useful for logging/orientation checks, not for grasp choice.
    """
    mask_full = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_full, [cnt], -1, 255, thickness=-1)
    dilated = cv2.dilate(mask_full, np.ones((9, 9), np.uint8))
    ring = cv2.subtract(dilated, mask_full)

    lo = np.array(config.CHIP_BORDER_HSV_RANGE[0])
    hi = np.array(config.CHIP_BORDER_HSV_RANGE[1])
    border_mask = cv2.inRange(hsv_image, lo, hi)
    ring_border = cv2.bitwise_and(border_mask, ring)

    ring_pixels = cv2.countNonZero(ring)
    if ring_pixels == 0:
        return False
    return cv2.countNonZero(ring_border) / ring_pixels > 0.3
