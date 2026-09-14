"""
vision/camera_interface.py
Thin wrapper around an Intel RealSense 3D camera. Swap the pyrealsense2
calls for your own SDK if you use a different camera; everything
downstream only depends on the methods defined on this class.
"""

import numpy as np
import pyrealsense2 as rs
import config


class CameraInterface:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                           rs.format.z16, config.CAMERA_FPS)
        cfg.enable_stream(rs.stream.color, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                           rs.format.bgr8, config.CAMERA_FPS)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        """Returns (color_image BGR uint8 ndarray, depth_frame rs.frame, depth_image uint16 ndarray)."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_frame, depth_image

    def pixel_to_point(self, depth_frame, px, py):
        """3D point (meters) in the camera's optical frame for pixel (px, py)."""
        depth = depth_frame.get_distance(int(px), int(py))
        if depth <= 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(px), float(py)], depth)
        return np.array(point)

    def local_surface_normal(self, depth_frame, px, py, patch_radius_px=6):
        """
        Estimates the local surface normal at pixel (px, py) by fitting a
        plane (via SVD) to a small neighborhood of 3D points. This lets the
        arm align its approach with a chip's actual face orientation even
        when chips rest at an angle on an uneven pile, rather than always
        approaching along world Z.
        """
        pts = []
        h, w = self.intrinsics.height, self.intrinsics.width
        for dy in range(-patch_radius_px, patch_radius_px + 1, 2):
            for dx in range(-patch_radius_px, patch_radius_px + 1, 2):
                x, y = int(px + dx), int(py + dy)
                if 0 <= x < w and 0 <= y < h:
                    p = self.pixel_to_point(depth_frame, x, y)
                    if p is not None and config.DEPTH_MIN_VALID < p[2] < config.DEPTH_MAX_VALID:
                        pts.append(p)
        if len(pts) < 6:
            return None
        pts = np.array(pts)
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2]
        if normal[2] > 0:          # keep normal pointing back toward the camera
            normal = -normal
        return normal / np.linalg.norm(normal)

    def stop(self):
        self.pipeline.stop()
