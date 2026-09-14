"""
depth_viewer.py
---------------
Displays the color image and prints the depth value at the center (or any clicked pixel).
Press 'q' to quit.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
pipeline.start(config)
align = rs.align(rs.stream.color)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_m = depth_frame.get_distance(x, y)
        print(f"Depth at ({x},{y}): {depth_m:.3f} m")

cv2.namedWindow("Depth Viewer")
cv2.setMouseCallback("Depth Viewer", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Show depth value at center
        center_depth = depth_frame.get_distance(320, 240)
        cv2.putText(color_image, f"Center depth: {center_depth:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Depth Viewer", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()