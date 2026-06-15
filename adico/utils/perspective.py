"""
utils/perspective.py — Corner selection and perspective warp utilities.
"""

import cv2
import numpy as np
from typing import Optional


def load_corners(path: str) -> Optional[np.ndarray]:
    """
    Loads saved (4,2) float32 array from .npy file.

    Args:
        path (str): Path to the .npy file.

    Returns:
        Optional[np.ndarray]: The loaded corners array or None if not found.
    """
    try:
        return np.load(path).astype(np.float32)
    except FileNotFoundError:
        return None


def select_corners(img: np.ndarray, save_path: str) -> np.ndarray:
    """
    Shows image, user clicks TL→TR→BR→BL. Uses OpenCV mouse callback.
    Saves to .npy and returns (4,2) float32 array.

    Args:
        img (np.ndarray): The input image.
        save_path (str): Path to save the corners.

    Returns:
        np.ndarray: The selected corners (4,2).
    """
    corners = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Corners", img)

    cv2.imshow("Select Corners", img)
    cv2.setMouseCallback("Select Corners", mouse_callback)

    while len(corners) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow("Select Corners")

    corners = np.array(corners, dtype=np.float32)
    np.save(save_path, corners)
    return corners


def compute_homography(corners: np.ndarray, out_px: int) -> np.ndarray:
    """
    Computes and returns the 3×3 homography matrix from corners to a square output.

    Args:
        corners (np.ndarray): The (4,2) corners array.
        out_px (int): The output size in pixels.

    Returns:
        np.ndarray: The 3x3 homography matrix.
    """
    dst = np.array([
        [0, 0],
        [out_px, 0],
        [out_px, out_px],
        [0, out_px]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst)
    return H


def warp_frame(frame: np.ndarray, H: np.ndarray, out_px: int) -> np.ndarray:
    """
    Applies cv2.warpPerspective and returns the warped top-down image.

    Args:
        frame (np.ndarray): The input frame.
        H (np.ndarray): The homography matrix.
        out_px (int): The output size in pixels.

    Returns:
        np.ndarray: The warped image.
    """
    return cv2.warpPerspective(frame, H, (out_px, out_px))