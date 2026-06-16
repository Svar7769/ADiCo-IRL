# camera.py — Thin wrapper around cv2.VideoCapture for webcam frame capture.
import time
import cv2
import numpy as np


class Camera:
    def __init__(self, config: dict):
        # Open camera with the V4L2 backend (needed for format control on Linux).
        self.cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {config['camera_index']}")

        # Force MJPEG before resolution — the C920 only does 720p30 over MJPEG;
        # raw YUYV crawls at ~7-10 fps and stalls the control loop.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up — sleep then discard first 30 frames to flush buffer.
        time.sleep(config["camera_warmup_s"])
        for _ in range(30):
            self.cap.read()

    def get_frame(self) -> np.ndarray:
        # Grab the freshest frame, discarding buffered backlog so the control
        # loop always sees current robot positions, not stale ones.
        for _ in range(4):
            self.cap.grab()
        while True:
            ret, frame = self.cap.retrieve()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            if np.mean(frame) >= 5:
                return frame

    def release(self):
        self.cap.release()