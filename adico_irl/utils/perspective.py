# perspective.py — Corner selection and perspective warp utilities.
# Converts a tilted camera view of the arena into a flat top-down square image.

import cv2
import numpy as np
from typing import Optional


def load_corners(path: str) -> Optional[np.ndarray]:
    # Load saved (4,2) corner array from .npy file. Returns None if file not found.
    try:
        return np.load(path).astype(np.float32)
    except FileNotFoundError:
        return None


def select_corners(img: np.ndarray, save_path: str) -> np.ndarray:
    # Show image and let user click TL → TR → BR → BL.
    # Draws a green dot on each click. Saves and returns (4,2) float32 array.
    corners = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Corners", img)

    cv2.imshow("Select Corners", img)
    cv2.setMouseCallback("Select Corners", mouse_callback)

    # Wait until all 4 corners are clicked.
    while len(corners) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow("Select Corners")

    corners = np.array(corners, dtype=np.float32)
    np.save(save_path, corners)
    return corners


def compute_homography(corners: np.ndarray, out_px: int) -> np.ndarray:
    # Compute 3x3 homography matrix that maps the clicked corners to a perfect square.
    dst = np.array([
        [0,      0     ],   # TL
        [out_px, 0     ],   # TR
        [out_px, out_px],   # BR
        [0,      out_px],   # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners, dst)
    return H


def warp_frame(frame: np.ndarray, H: np.ndarray, out_px: int) -> np.ndarray:
    # Apply homography H to produce a top-down out_px × out_px image.
    return cv2.warpPerspective(frame, H, (out_px, out_px))