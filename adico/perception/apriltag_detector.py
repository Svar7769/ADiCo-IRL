"""
apriltag_detector.py — Detects robots and goals by AprilTag ID in the warped top-down image.
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, Optional


class AprilTagDetector:
    def __init__(self, config: dict):
        adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        # Wider adaptive threshold window catches tags at more scales/lighting conditions
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        # Allow smaller markers (default 0.03 is too strict for overhead small tags)
        params.minMarkerPerimeterRate = 0.02
        # More lenient corner refinement
        params.polygonalApproxAccuracyRate = 0.08
        self._detector = cv2.aruco.ArucoDetector(adict, params)

        self.arena_size_m = config["arena_size_m"]
        self.warped_px = config["warped_px"]
        self._n_agents = config["n_agents"]

        # tag_id -> (agent_id, "robot"|"goal")
        self._tag_map: Dict[int, tuple] = {}
        for agent in config["agents"]:
            self._tag_map[agent["robot_tag"]] = (agent["id"], "robot")
            self._tag_map[agent["goal_tag"]] = (agent["id"], "goal")

    def detect(self, warped: np.ndarray) -> Dict[str, Any]:
        corners_list, ids, _ = self._detector.detectMarkers(warped)

        robots: list = [None] * self._n_agents
        goals: list = [None] * self._n_agents
        headings: list = [None] * self._n_agents

        if ids is not None:
            for corners, tag_id in zip(corners_list, ids.flatten()):
                if tag_id not in self._tag_map:
                    continue
                agent_id, tag_type = self._tag_map[tag_id]
                pts = corners[0]  # (4,2): TL, TR, BR, BL
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                x_m = (cx - self.warped_px / 2) * (self.arena_size_m / self.warped_px)
                y_m = (self.warped_px / 2 - cy) * (self.arena_size_m / self.warped_px)
                pos = np.array([x_m, y_m])

                if tag_type == "robot":
                    robots[agent_id] = pos
                    # Heading: centroid → midpoint of top edge (TL+TR)/2, flipped to world frame
                    mid_top = (pts[0] + pts[1]) / 2.0
                    headings[agent_id] = math.atan2(-(mid_top[1] - cy), mid_top[0] - cx)
                else:
                    goals[agent_id] = pos

        return {
            "agents": [
                {
                    "id": i,
                    "robot_pos": robots[i],
                    "goal_pos": goals[i],
                    "robot_detected": robots[i] is not None,
                    "goal_detected": goals[i] is not None,
                    "robot_heading": headings[i],
                }
                for i in range(self._n_agents)
            ]
        }
