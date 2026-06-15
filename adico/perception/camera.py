"""
camera.py — Thin wrapper around cv2.VideoCapture for webcam frame capture.
"""

import time
import cv2
import numpy as np


class Camera:
    """
    Thin wrapper around cv2.VideoCapture for webcam frame capture.

    Attributes:
        cap (cv2.VideoCapture): The OpenCV video capture object.
    """

    def __init__(self, config: dict):
        """
        Initializes the camera.

        Args:
            config (dict): Configuration dictionary containing camera settings.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(config["camera_index"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {config['camera_index']}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])

        # Warm up the camera
        time.sleep(config["camera_warmup_s"])
        for _ in range(30):
            self.cap.read()  # Discard warmup frames

    def get_frame(self) -> np.ndarray:
        """
        Captures and returns one BGR frame from the camera.

        Skips blank frames (mean pixel value < 5).

        Returns:
            np.ndarray: The captured BGR frame.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            if np.mean(frame) >= 5:
                return frame

    def release(self):
        """
        Releases the camera capture.
        """
        self.cap.release()