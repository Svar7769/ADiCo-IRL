"""
main.py — The orchestrator. Ties everything together in a 10 Hz control loop.
"""

import argparse
import logging
import sys
import time
import yaml
import numpy as np

from adico_irl.perception.camera import Camera
from adico_irl.perception.apriltag_detector import AprilTagDetector
from adico_irl.perception.observation_builder import ObservationBuilder
from adico_irl.policy.policy_runner import PolicyRunner
from adico_irl.control.movement_controller import MultiVectorController
from adico_irl.utils.perspective import load_corners, compute_homography, warp_frame
from adico_irl.utils.visualization import DebugVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="ADiCo Vector Deployment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no-robot", action="store_true", help="Run without robots for testing")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Log level")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize modules
    camera = Camera(config)
    corners = load_corners(config["corners_path"])
    if corners is None:
        logging.error("Run calibrate.py first!")
        sys.exit(1)
    H = compute_homography(corners, config["warped_px"])
    detector = AprilTagDetector(config)
    obs_builder = ObservationBuilder(config)
    policy = PolicyRunner(config)
    if not args.no_robot:
        controller = MultiVectorController(config)
    else:
        controller = None
    viz = DebugVisualizer(config) if not args.no_viz else None

    dt = 1.0 / config["control_hz"]

    logging.info("Starting control loop")

    try:
        while True:
            t0 = time.time()

            frame = camera.get_frame()
            warped = warp_frame(frame, H, config["warped_px"])
            detections = detector.detect(warped)
            obs = obs_builder.update(detections)

            if obs is not None:
                actions = policy.get_actions(obs)
                goal_thresh = config.get("goal_reached_threshold", 0.1)
                for i in range(len(actions)):
                    if np.hypot(obs[i][4], obs[i][5]) < goal_thresh:
                        actions[i] = np.zeros(config["action_dim"])
                if controller:
                    controller.execute(actions)
                else:
                    # Print actions
                    action_str = " ".join(f"R{i}=({actions[i][0]:.2f}, {actions[i][1]:.2f})" for i in range(len(actions)))
                    logging.info(f"[main] Actions: {action_str}")
            else:
                if controller:
                    controller.stop_all()
                logging.warning("[main] Missing detections — holding position")

            if viz:
                actions_viz = actions if obs is not None else np.zeros((config["n_agents"], 2))
                if viz.update(warped, detections, actions_viz):
                    break  # user pressed 'q'

            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        if controller:
            controller.stop_all()
            controller.disconnect()
        camera.release()
        if viz:
            viz.close()


if __name__ == "__main__":
    main()