"""
live_grid_detection.py
----------------------
Dynamic Connect 4 grid detection from live camera feed.
Press 'c' to freeze frame and click corners (TL, TR, BR, BL).
Once corners are set, the grid is warped and classified in real time.
Press 'q' to quit.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# ---------- CAMERA SETUP ----------
def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().first_color_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 1.0)
    return pipeline

def get_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())

# ---------- HSV RANGES (adjust via trackbars) ----------
def create_trackbars():
    cv2.namedWindow("Controls")
    cv2.createTrackbar("R_Low_H", "Controls", 0, 179, lambda x: None)
    cv2.createTrackbar("R_Low_S", "Controls", 80, 255, lambda x: None)
    cv2.createTrackbar("R_Low_V", "Controls", 65, 255, lambda x: None)
    cv2.createTrackbar("R_High_H", "Controls", 19, 179, lambda x: None)
    cv2.createTrackbar("R_High_S", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("R_High_V", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("R2_Low_H", "Controls", 170, 179, lambda x: None)
    cv2.createTrackbar("R2_High_H", "Controls", 179, 179, lambda x: None)
    cv2.createTrackbar("Y_Low_H", "Controls", 20, 179, lambda x: None)
    cv2.createTrackbar("Y_Low_S", "Controls", 60, 255, lambda x: None)
    cv2.createTrackbar("Y_Low_V", "Controls", 50, 255, lambda x: None)
    cv2.createTrackbar("Y_High_H", "Controls", 90, 179, lambda x: None)
    cv2.createTrackbar("Y_High_S", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("Y_High_V", "Controls", 255, 255, lambda x: None)

def get_hsv_ranges():
    r_low_h = cv2.getTrackbarPos("R_Low_H", "Controls")
    r_low_s = cv2.getTrackbarPos("R_Low_S", "Controls")
    r_low_v = cv2.getTrackbarPos("R_Low_V", "Controls")
    r_high_h = cv2.getTrackbarPos("R_High_H", "Controls")
    r_high_s = cv2.getTrackbarPos("R_High_S", "Controls")
    r_high_v = cv2.getTrackbarPos("R_High_V", "Controls")
    r2_low_h = cv2.getTrackbarPos("R2_Low_H", "Controls")
    r2_high_h = cv2.getTrackbarPos("R2_High_H", "Controls")
    y_low_h = cv2.getTrackbarPos("Y_Low_H", "Controls")
    y_low_s = cv2.getTrackbarPos("Y_Low_S", "Controls")
    y_low_v = cv2.getTrackbarPos("Y_Low_V", "Controls")
    y_high_h = cv2.getTrackbarPos("Y_High_H", "Controls")
    y_high_s = cv2.getTrackbarPos("Y_High_S", "Controls")
    y_high_v = cv2.getTrackbarPos("Y_High_V", "Controls")
    return {
        'red1': (np.array([r_low_h, r_low_s, r_low_v]), np.array([r_high_h, r_high_s, r_high_v])),
        'red2': (np.array([r2_low_h, r_low_s, r_low_v]), np.array([r2_high_h, r_high_s, r_high_v])),
        'yellow': (np.array([y_low_h, y_low_s, y_low_v]), np.array([y_high_h, y_high_s, y_high_v])),
    }

def classify_cell(cell_img, ranges):
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, ranges['red1'][0], ranges['red1'][1])
    mask_red2 = cv2.inRange(hsv, ranges['red2'][0], ranges['red2'][1])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, ranges['yellow'][0], ranges['yellow'][1])
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    total = cell_img.shape[0] * cell_img.shape[1]
    threshold = 0.02 * total
    if red_pixels > threshold and red_pixels > yellow_pixels:
        return 1  # red
    elif yellow_pixels > threshold and yellow_pixels > red_pixels:
        return 2  # yellow
    else:
        return 0  # empty

# ---------- MOUSE CALLBACK ----------
corners = []
def mouse_callback(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            print(f"Corner {len(corners)}: ({x}, {y})")

# ---------- MAIN ----------
def main():
    global corners
    pipeline = init_camera()
    create_trackbars()

    cv2.namedWindow("Live Grid Detection")
    cv2.setMouseCallback("Live Grid Detection", mouse_callback)

    # State
    captured_img = None
    corners_set = False
    warp_matrix = None
    out_w, out_h = 700, 600
    rows, cols = 6, 7
    print("Press 'c' to freeze frame and click the four corners.")
    print("Press 'r' to reset corners, 'q' to quit.")
    print("Once corners are set, processing will be live.")

    while True:
        img = get_frame(pipeline)
        if img is None:
            continue

        # If we have corners, use them to warp and classify live
        if len(corners) == 4:
            # Compute warp matrix once
            if not corners_set:
                pts_src = np.array(corners, dtype=np.float32)
                pts_dst = np.array([
                    [0, 0],
                    [out_w - 1, 0],
                    [out_w - 1, out_h - 1],
                    [0, out_h - 1]
                ], dtype=np.float32)
                warp_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                corners_set = True
                print("Corners set. Live processing started.")

            # Warp the current frame
            warped = cv2.warpPerspective(img, warp_matrix, (out_w, out_h))
            # Create overlay
            overlay = warped.copy()
            cell_h = out_h // rows
            cell_w = out_w // cols
            margin = 5
            board_state = np.zeros((rows, cols), dtype=np.int32)
            ranges = get_hsv_ranges()

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
                    cell_class = classify_cell(cell_img, ranges)
                    board_state[r, c] = cell_class
                    color = (255, 255, 255)
                    if cell_class == 1:
                        color = (0, 0, 255)     # red
                    elif cell_class == 2:
                        color = (0, 255, 255)   # yellow
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Optional: show class number
                    # cv2.putText(overlay, str(cell_class), (x1+5, y1+20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Combine original warped and overlay
            combined = np.hstack((warped, overlay))
            cv2.imshow("Live Grid Detection", combined)
            # Print board state every 30 frames
            if hasattr(main, 'frame_counter'):
                main.frame_counter += 1
            else:
                main.frame_counter = 0
            if main.frame_counter % 30 == 0:
                print("\nBoard state (0=empty, 1=red, 2=yellow):")
                for r in range(rows):
                    print(" ".join(str(board_state[r, c]) for c in range(cols)))
        else:
            # No corners set – show live feed with instruction
            if captured_img is not None:
                display = captured_img.copy()
            else:
                display = img.copy()
            # Draw any existing corners
            for pt in corners:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
            if len(corners) > 1:
                cv2.polylines(display, [np.array(corners, dtype=np.int32)], False, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to capture, click corners (TL,TR,BR,BL)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Live Grid Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture a frame for corner selection
            captured_img = get_frame(pipeline)
            if captured_img is not None:
                corners = []
                corners_set = False
                print("Frame captured. Click the four corners in order.")
        elif key == ord('r'):
            corners = []
            corners_set = False
            print("Corners reset.")

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()