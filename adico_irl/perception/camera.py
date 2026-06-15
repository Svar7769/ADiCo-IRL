# camera.py — Thin wrapper around cv2.VideoCapture for webcam frame capture.

import time
import cv2
import numpy as np


class Camera:

    def __init__(self, config: dict):
        # Open camera at the given index.
        self.cap = cv2.VideoCapture(config["camera_index"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {config['camera_index']}")

        # Request resolution from config.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])

        # Warm up — sleep then discard first 30 frames to flush buffer.
        time.sleep(config["camera_warmup_s"])
        for _ in range(30):
            self.cap.read()

    def get_frame(self) -> np.ndarray:
        # Capture and return one BGR frame. Skips blank frames (mean < 5).
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            if np.mean(frame) >= 5:
                return frame

    def release(self):
        # Release the camera.
        self.cap.release()