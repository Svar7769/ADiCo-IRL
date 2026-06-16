# apriltag_detector.py — Detects robots and goals by AprilTag ID in the warped top-down image.
import math
import logging
import cv2
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AprilTagDetector:
    def __init__(self, config: dict):
        # Use AprilTag 36h11 family.
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()

        # Wider threshold window — catches tags across different scales/lighting.
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 53
        params.adaptiveThreshWinSizeStep = 4

        # Allow small tags visible from the overhead camera.
        params.minMarkerPerimeterRate = 0.02

        # Lenient corner approximation for slightly warped tags.
        params.polygonalApproxAccuracyRate = 0.08

        # Sub-pixel corner refinement — big improvement for small/marginal tags.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector     = cv2.aruco.ArucoDetector(tag_dict, params)
        self.arena_size_m = config["arena_size_m"]
        self.warped_px    = config["warped_px"]
        self.n_agents     = config["n_agents"]

        # Build tag lookup table: tag_id -> (agent_id, "robot" | "goal")
        self.tag_map: Dict[int, tuple] = {}
        for agent in config["agents"]:
            self.tag_map[agent["robot_tag"]] = (agent["id"], "robot")
            self.tag_map[agent["goal_tag"]]  = (agent["id"], "goal")

        logger.info(f"[detector] expecting tag IDs: "
                    f"{sorted(self.tag_map.keys())}")

        # Throttle the per-frame debug print.
        self._frame_count = 0
        self._debug_every = 10   # log detected IDs once every N frames

    def detect(self, warped: np.ndarray) -> Dict[str, Any]:
        corners_list, ids, _ = self.detector.detectMarkers(warped)

        # Periodic diagnostic: what was seen vs. what we map.
        self._frame_count += 1
        if self._frame_count % self._debug_every == 0:
            seen = ids.flatten().tolist() if ids is not None else []
            mapped   = [t for t in seen if t in self.tag_map]
            unmapped = [t for t in seen if t not in self.tag_map]
            logger.info(f"[detector] seen={seen}  mapped={mapped}  "
                        f"ignored={unmapped}")

        robots   = [None] * self.n_agents
        goals    = [None] * self.n_agents
        headings = [None] * self.n_agents

        if ids is not None:
            for corners, tag_id in zip(corners_list, ids.flatten()):
                if tag_id not in self.tag_map:
                    continue
                agent_id, tag_type = self.tag_map[tag_id]

                # pts is (4,2) in order TL, TR, BR, BL.
                pts = corners[0]
                cx  = pts[:, 0].mean()
                cy  = pts[:, 1].mean()

                # Convert pixel center to meters relative to arena center.
                x_m = (cx - self.warped_px / 2) * (self.arena_size_m / self.warped_px)
                y_m = (self.warped_px / 2 - cy) * (self.arena_size_m / self.warped_px)
                pos = np.array([x_m, y_m])

                if tag_type == "robot":
                    robots[agent_id] = pos
                    # Heading: tag center toward midpoint of top edge (TL+TR).
                    mid_top = (pts[0] + pts[1]) / 2.0
                    headings[agent_id] = math.atan2(-(mid_top[1] - cy),
                                                    mid_top[0] - cx)
                else:
                    goals[agent_id] = pos

        return {
            "agents": [
                {
                    "id":             i,
                    "robot_pos":      robots[i],
                    "goal_pos":       goals[i],
                    "robot_detected": robots[i] is not None,
                    "goal_detected":  goals[i] is not None,
                    "robot_heading":  headings[i],
                }
                for i in range(self.n_agents)
            ]
        }