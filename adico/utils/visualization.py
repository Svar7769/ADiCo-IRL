"""
utils/visualization.py — Optional debug overlay for the live loop.
"""

import cv2
import numpy as np
from typing import Dict, Any


class DebugVisualizer:
    """
    Optional debug overlay for the live loop.

    Attributes:
        window_name (str): Name of the OpenCV window.
        warped_px (int): Warped image size.
        arena_size_m (float): Arena size in meters.
    """

    def __init__(self, config: dict):
        """
        Initializes the visualizer.

        Args:
            config (dict): Configuration dictionary.
        """
        self.window_name = "ADiCo Vector Debug"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.warped_px = config["warped_px"]
        self.arena_size_m = config["arena_size_m"]

    def update(self, warped: np.ndarray, detections: Dict[str, Any], actions: np.ndarray) -> bool:
        """
        Draws on the warped image and shows it.

        Args:
            warped (np.ndarray): The warped image.
            detections (dict): Detection results.
            actions (np.ndarray): Actions (n_agents, 2).

        Returns:
            bool: True if user pressed 'q'.
        """
        img = warped.copy()

        for agent in detections["agents"]:
            i = agent["id"]
            color = (255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255)

            # Robot position
            if agent["robot_detected"]:
                px = int((agent["robot_pos"][0] / self.arena_size_m + 0.5) * self.warped_px)
                py = int((0.5 - agent["robot_pos"][1] / self.arena_size_m) * self.warped_px)
                cv2.circle(img, (px, py), 10, color, -1)
                cv2.putText(img, f"R{i}", (px-10, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Heading arrow
                if agent.get("robot_heading") is not None:
                    import math
                    hx = int(px + 20 * math.cos(agent["robot_heading"]))
                    hy = int(py - 20 * math.sin(agent["robot_heading"]))
                    cv2.arrowedLine(img, (px, py), (hx, hy), (255, 255, 255), 2, tipLength=0.4)

                # Action arrow
                vx, vy = actions[i]
                if np.hypot(vx, vy) > 0.01:
                    scale = 50  # pixels
                    dx = int(vx / self.arena_size_m * scale)
                    dy = int(-vy / self.arena_size_m * scale)
                    cv2.arrowedLine(img, (px, py), (px + dx, py + dy), color, 2)

            # Goal position
            if agent["goal_detected"]:
                gx = int((agent["goal_pos"][0] / self.arena_size_m + 0.5) * self.warped_px)
                gy = int((0.5 - agent["goal_pos"][1] / self.arena_size_m) * self.warped_px)
                cv2.circle(img, (gx, gy), 8, color, 2)
                cv2.putText(img, f"G{i}", (gx-10, gy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Dashed line to robot
                if agent["robot_detected"]:
                    cv2.line(img, (px, py), (gx, gy), color, 1, cv2.LINE_AA)

        # FPS and positions text
        y_offset = 20
        for agent in detections["agents"]:
            i = agent["id"]
            text = f"R{i}: "
            if agent["robot_detected"]:
                text += f"({agent['robot_pos'][0]:.2f}, {agent['robot_pos'][1]:.2f})"
            else:
                text += "N/A"
            text += f" A:({actions[i][0]:.2f}, {actions[i][1]:.2f})"
            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')

    def close(self):
        """
        Closes the window.
        """
        cv2.destroyWindow(self.window_name)