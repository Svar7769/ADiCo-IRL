# visualization.py — Live debug overlay for the control loop.
# Draws robot positions, headings, actions, and goals on the warped arena image.

import math
import cv2
import numpy as np
from typing import Dict, Any

# Agent colors in BGR — blue, green, red.
AGENT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class DebugVisualizer:

    def __init__(self, config: dict):
        self.window_name  = "ADiCo Vector"
        self.warped_px    = config["warped_px"]
        self.arena_size_m = config["arena_size_m"]
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _to_px(self, x_m: float, y_m: float):
        # Convert world coordinates (meters) to image pixel coordinates.
        px = int((x_m / self.arena_size_m + 0.5) * self.warped_px)
        py = int((0.5 - y_m / self.arena_size_m) * self.warped_px)
        return px, py

    def update(self, warped: np.ndarray, detections: Dict[str, Any], actions: np.ndarray) -> bool:
        # Draw all overlays and show the frame. Returns True if user pressed 'q'.
        img = warped.copy()

        for agent in detections["agents"]:
            i     = agent["id"]
            color = AGENT_COLORS[i]

            if agent["robot_detected"]:
                px, py = self._to_px(*agent["robot_pos"])

                # Filled circle and label for robot position.
                cv2.circle(img, (px, py), 10, color, -1)
                cv2.putText(img, f"R{i}", (px-10, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # White arrow showing which way the robot is facing.
                if agent.get("robot_heading") is not None:
                    hx = int(px + 20 * math.cos(agent["robot_heading"]))
                    hy = int(py - 20 * math.sin(agent["robot_heading"]))
                    cv2.arrowedLine(img, (px, py), (hx, hy), (255, 255, 255), 2, tipLength=0.4)

                # Colored arrow showing where the policy wants the robot to move.
                vx, vy = actions[i]
                if np.hypot(vx, vy) > 0.01:
                    scale = 50
                    dx    = int(vx / self.arena_size_m * scale)
                    dy    = int(-vy / self.arena_size_m * scale)
                    cv2.arrowedLine(img, (px, py), (px + dx, py + dy), color, 2)

            if agent["goal_detected"]:
                gx, gy = self._to_px(*agent["goal_pos"])

                # Hollow circle and label for goal position.
                cv2.circle(img, (gx, gy), 8, color, 2)
                cv2.putText(img, f"G{i}", (gx-10, gy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Line connecting robot to its goal.
                if agent["robot_detected"]:
                    cv2.line(img, (px, py), (gx, gy), color, 1, cv2.LINE_AA)

        # Text overlay — one line per agent showing position and current action.
        y = 20
        for agent in detections["agents"]:
            i    = agent["id"]
            pos  = f"({agent['robot_pos'][0]:.2f}, {agent['robot_pos'][1]:.2f})" if agent["robot_detected"] else "N/A"
            text = f"R{i}: {pos}  A:({actions[i][0]:.2f}, {actions[i][1]:.2f})"
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 15

        cv2.imshow(self.window_name, img)
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close(self):
        cv2.destroyWindow(self.window_name)