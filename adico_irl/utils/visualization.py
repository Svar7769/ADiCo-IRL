# visualization.py — Live debug overlay for the control loop.
# Window 1 (OpenCV): warped arena with robots, headings, actions, goals.
# Window 2 (matplotlib): XY position trails, styled to match the demo's
#                        trajectory panel.

import math
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interactive backend; swap to Qt5Agg if no tkinter
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, Any

AGENT_COLORS_BGR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]   # blue, green, red
AGENT_COLORS_MPL = ["blue", "green", "red"]

TRAIL_LEN    = 300     # position points kept per robot
RENDER_EVERY = 5       # redraw matplotlib every N ticks (throttle for speed)


class DebugVisualizer:
    def __init__(self, config: dict):
        self.window_name  = "ADiCo Vector"
        self.warped_px    = config["warped_px"]
        self.arena_size_m = config["arena_size_m"]
        self.n_agents     = config["n_agents"]

        self.trails  = [deque(maxlen=TRAIL_LEN) for _ in range(self.n_agents)]
        self.goal_xy = [None] * self.n_agents
        self._tick   = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # --- matplotlib position window (styled like the demo's XY panel) ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.manager.set_window_title("ADiCo Positions")
        self.ax.set(title="XY Trajectory (m)", xlabel="x", ylabel="y",
                    xlim=(-1, 1), ylim=(-1, 1))
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # Per-robot handles: trail+marker line (demo's "-o" style) and goal star.
        self.trail_lines = []
        self.goal_dots   = []
        for i in range(self.n_agents):
            c = AGENT_COLORS_MPL[i]
            (tl,) = self.ax.plot([], [], "-o", color=c, ms=2, lw=1,
                                 label=f"R{i}")
            (gd,) = self.ax.plot([], [], "*", color=c, ms=14,
                                 markeredgecolor="black", markeredgewidth=0.5,
                                 label=f"G{i}")
            self.trail_lines.append(tl)
            self.goal_dots.append(gd)
        self.ax.legend(fontsize=7)

    def _to_px(self, x_m, y_m):
        px = int((x_m / self.arena_size_m + 0.5) * self.warped_px)
        py = int((0.5 - y_m / self.arena_size_m) * self.warped_px)
        return px, py

    def update(self, warped: np.ndarray, detections: Dict[str, Any],
               actions: np.ndarray) -> bool:
        # --- record history ---
        for agent in detections["agents"]:
            i = agent["id"]
            if agent["robot_detected"]:
                self.trails[i].append(tuple(agent["robot_pos"]))
            if agent["goal_detected"]:
                self.goal_xy[i] = tuple(agent["goal_pos"])

        # --- OpenCV arena window ---
        img = warped.copy()
        for agent in detections["agents"]:
            i     = agent["id"]
            color = AGENT_COLORS_BGR[i]
            if agent["robot_detected"]:
                px, py = self._to_px(*agent["robot_pos"])
                cv2.circle(img, (px, py), 10, color, -1)
                cv2.putText(img, f"R{i}", (px - 10, py - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if agent.get("robot_heading") is not None:
                    hx = int(px + 20 * math.cos(agent["robot_heading"]))
                    hy = int(py - 20 * math.sin(agent["robot_heading"]))
                    cv2.arrowedLine(img, (px, py), (hx, hy),
                                    (255, 255, 255), 2, tipLength=0.4)
                vx, vy = actions[i]
                if np.hypot(vx, vy) > 0.01:
                    dx = int(vx / self.arena_size_m * 50)
                    dy = int(-vy / self.arena_size_m * 50)
                    cv2.arrowedLine(img, (px, py), (px + dx, py + dy), color, 2)
            if agent["goal_detected"]:
                gx, gy = self._to_px(*agent["goal_pos"])
                cv2.circle(img, (gx, gy), 8, color, 2)
                cv2.putText(img, f"G{i}", (gx - 10, gy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if agent["robot_detected"]:
                    cv2.line(img, (px, py), (gx, gy), color, 1, cv2.LINE_AA)

        y = 20
        for agent in detections["agents"]:
            i   = agent["id"]
            pos = (f"({agent['robot_pos'][0]:.2f},{agent['robot_pos'][1]:.2f})"
                   if agent["robot_detected"] else "N/A")
            cv2.putText(img, f"R{i}: {pos}  A:({actions[i][0]:.2f},{actions[i][1]:.2f})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 15

        cv2.imshow(self.window_name, img)

        # --- matplotlib position window (throttled) ---
        self._tick += 1
        if self._tick % RENDER_EVERY == 0:
            for i in range(self.n_agents):
                if self.trails[i]:
                    xs = [p[0] for p in self.trails[i]]
                    ys = [p[1] for p in self.trails[i]]
                    self.trail_lines[i].set_data(xs, ys)
                if self.goal_xy[i] is not None:
                    self.goal_dots[i].set_data([self.goal_xy[i][0]],
                                               [self.goal_xy[i][1]])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        return cv2.waitKey(1) & 0xFF == ord('q')

    def close(self):
        cv2.destroyWindow(self.window_name)
        plt.close(self.fig)