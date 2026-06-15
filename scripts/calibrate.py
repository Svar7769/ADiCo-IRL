"""
calibrate.py — One-time interactive calibration: select arena corners and verify AprilTag detection.
"""

import sys
import cv2
import numpy as np
import yaml

from adico.perception.camera import Camera
from adico.perception.apriltag_detector import AprilTagDetector
from adico.utils.perspective import select_corners, compute_homography, warp_frame


def main():
    if len(sys.argv) != 2:
        print("Usage: python calibrate.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    camera = Camera(config)
    frame = camera.get_frame()
    camera.release()

    # Step 1: select arena corners
    corners = select_corners(frame.copy(), config["corners_path"])
    H = compute_homography(corners, config["warped_px"])
    warped = warp_frame(frame, H, config["warped_px"])

    # Step 2: verify all 6 AprilTags are visible
    _verify_tags(warped, config)


def _verify_tags(warped: np.ndarray, config: dict):
    detector = AprilTagDetector(config)
    detections = detector.detect(warped)

    img = warped.copy()
    agent_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    all_ok = True
    for agent in detections["agents"]:
        i = agent["id"]
        color = agent_colors[i]

        if agent["robot_detected"]:
            px = int((agent["robot_pos"][0] / config["arena_size_m"] + 0.5) * config["warped_px"])
            py = int((0.5 - agent["robot_pos"][1] / config["arena_size_m"]) * config["warped_px"])
            cv2.circle(img, (px, py), 12, color, -1)
            cv2.putText(img, f"R{i}", (px - 10, py - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  Robot {i}: OK  ({agent['robot_pos'][0]:.3f}, {agent['robot_pos'][1]:.3f}) m")
        else:
            print(f"  Robot {i}: NOT DETECTED")
            all_ok = False

        if agent["goal_detected"]:
            gx = int((agent["goal_pos"][0] / config["arena_size_m"] + 0.5) * config["warped_px"])
            gy = int((0.5 - agent["goal_pos"][1] / config["arena_size_m"]) * config["warped_px"])
            cv2.circle(img, (gx, gy), 10, color, 2)
            cv2.putText(img, f"G{i}", (gx - 10, gy - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  Goal  {i}: OK  ({agent['goal_pos'][0]:.3f}, {agent['goal_pos'][1]:.3f}) m")
        else:
            print(f"  Goal  {i}: NOT DETECTED")
            all_ok = False

    status = "ALL TAGS FOUND" if all_ok else "SOME TAGS MISSING — check placement/lighting"
    cv2.putText(img, status, (10, config["warped_px"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if all_ok else (0, 0, 255), 2)

    print(f"\n{status}")
    print("Press any key to close.")
    cv2.imshow("AprilTag Verification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
