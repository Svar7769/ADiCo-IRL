# calibrate.py — One-time setup: select arena corners and verify all AprilTags are visible.
#
# Usage:
#   python scripts/calibrate.py config.yaml

import sys
import cv2
import numpy as np
import yaml

from adico_irl.perception.camera import Camera
from adico_irl.perception.apriltag_detector import AprilTagDetector
from adico_irl.utils.perspective import select_corners, compute_homography, warp_frame

AGENT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/calibrate.py <config_path>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Grab one frame from the camera then release it.
    camera = Camera(config)
    frame  = camera.get_frame()
    camera.release()

    # Step 1 — click the 4 arena corners to set up the perspective warp.
    corners = select_corners(frame.copy(), config["corners_path"])
    H       = compute_homography(corners, config["warped_px"])
    warped  = warp_frame(frame, H, config["warped_px"])

    # Step 2 — verify all robots and goals are visible in the warped image.
    verify_tags(warped, config)


def to_px(x_m, y_m, config):
    # Convert world meters to pixel coordinates in the warped image.
    px = int((x_m / config["arena_size_m"] + 0.5) * config["warped_px"])
    py = int((0.5 - y_m / config["arena_size_m"]) * config["warped_px"])
    return px, py


def verify_tags(warped: np.ndarray, config: dict):
    detector   = AprilTagDetector(config)
    detections = detector.detect(warped)
    img        = warped.copy()
    all_ok     = True

    for agent in detections["agents"]:
        i     = agent["id"]
        color = AGENT_COLORS[i]

        if agent["robot_detected"]:
            px, py = to_px(*agent["robot_pos"], config)
            cv2.circle(img, (px, py), 12, color, -1)
            cv2.putText(img, f"R{i}", (px-10, py-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  Robot {i}: OK  ({agent['robot_pos'][0]:.3f}, {agent['robot_pos'][1]:.3f}) m")
        else:
            print(f"  Robot {i}: NOT DETECTED")
            all_ok = False

        if agent["goal_detected"]:
            gx, gy = to_px(*agent["goal_pos"], config)
            cv2.circle(img, (gx, gy), 10, color, 2)
            cv2.putText(img, f"G{i}", (gx-10, gy-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  Goal  {i}: OK  ({agent['goal_pos'][0]:.3f}, {agent['goal_pos'][1]:.3f}) m")
        else:
            print(f"  Goal  {i}: NOT DETECTED")
            all_ok = False

    # Status banner at the bottom of the image.
    status       = "ALL TAGS FOUND" if all_ok else "SOME TAGS MISSING — check placement/lighting"
    banner_color = (0, 255, 0) if all_ok else (0, 0, 255)
    cv2.putText(img, status, (10, config["warped_px"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)

    print(f"\n{status}")
    print("Press any key to close.")
    cv2.imshow("AprilTag Verification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()