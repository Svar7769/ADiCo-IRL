import argparse
import logging
import sys
import os
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adico_irl.perception.camera import Camera
from adico_irl.perception.apriltag_detector import AprilTagDetector
from adico_irl.perception.observation_builder import ObservationBuilder
from adico_irl.policy.policy_runner import PolicyRunner
from adico_irl.control.movement_controller import MultiVectorController
from adico_irl.utils.perspective import load_corners, compute_homography, warp_frame
from adico_irl.utils.visualization import DebugVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def main():
    parser = argparse.ArgumentParser(description="ADiCo Vector Deployment")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--no-robot", action="store_true",
                        help="Run without robots for testing")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"],
                        default="INFO", help="Log level")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize modules.
    camera = Camera(config)

    corners = load_corners(config["corners_path"])
    if corners is None:
        logging.error("Run calibrate.py first!")
        sys.exit(1)
    H = compute_homography(corners, config["warped_px"])

    detector    = AprilTagDetector(config)
    obs_builder = ObservationBuilder(config)
    policy      = PolicyRunner(config)

    controller = None if args.no_robot else MultiVectorController(config)
    viz        = None if args.no_viz   else DebugVisualizer(config)

    # Loop / experiment parameters.
    dt             = 1.0 / config["control_hz"]
    n_agents       = config["n_agents"]
    action_dim     = config["action_dim"]
    goal_thresh    = config.get("goal_reached_threshold", 0.10)   # arrive -> stop
    release_thresh = config.get("goal_release_threshold", 0.25)   # goal moved -> re-engage
    obs_timeout    = config.get("obs_timeout_s", 0.5)             # watchdog window

    # --- Arena bounds (stop robots that leave the arena) ---
    arena_size   = config["arena_size_m"]
    arena_margin = config.get("arena_margin_m", 0.05)
    if config.get("arena_origin", "corner") == "center":
        bound_lo = -arena_size / 2.0 + arena_margin
        bound_hi =  arena_size / 2.0 - arena_margin
    else:  # corner origin: 0 .. arena_size
        bound_lo = arena_margin
        bound_hi = arena_size - arena_margin

    logging.info("Starting control loop")
    logging.info(f"goal_thresh={goal_thresh:.2f} m, "
                 f"release_thresh={release_thresh:.2f} m")
    logging.info(f"arena bounds: [{bound_lo:.2f}, {bound_hi:.2f}] m "
                 f"({config.get('arena_origin', 'corner')} origin)")

    actions       = None
    last_good_obs = time.time()
    reached       = [False] * n_agents   # per-robot goal-reached latch

    try:
        while True:
            t0 = time.time()

            # --- Perception ---
            frame      = camera.get_frame()
            warped     = warp_frame(frame, H, config["warped_px"])
            detections = detector.detect(warped)
            obs        = obs_builder.update(detections)

            if obs is not None: 
                # Temp 
                logging.info(f"R0 pos: {obs[0][0]:.2f}, {obs[0][1]:.2f}")
                
                last_good_obs = time.time()

                # --- Policy ---
                actions = policy.get_actions(obs)
                assert actions.shape[0] == len(obs), \
                    f"action/obs count mismatch: {actions.shape[0]} vs {len(obs)}"

                # --- Arena bounds safety ---
                # obs[i][0:2] is the robot's absolute arena position (meters).
                # Any robot outside the safe bounds is stopped immediately,
                # before the goal latch, so out-of-bounds always wins.
                out_of_bounds = [False] * n_agents
                for i in range(len(actions)):
                    px, py = obs[i][0], obs[i][1]
                    if not (bound_lo <= px <= bound_hi and
                            bound_lo <= py <= bound_hi):
                        out_of_bounds[i] = True
                        actions[i] = np.zeros(action_dim)
                        logging.warning(f"[main] R{i} out of bounds "
                                        f"({px:.2f}, {py:.2f}) — stopped")

                # --- Per-robot goal latch with hysteresis ---
                # Skipped for out-of-bounds robots (already zeroed above).
                for i in range(len(actions)):
                    if out_of_bounds[i]:
                        continue
                    goal_dist = np.hypot(obs[i][4], obs[i][5])

                    if reached[i]:
                        if goal_dist > release_thresh:
                            reached[i] = False                       # goal moved
                            logging.info(f"[main] R{i} goal moved "
                                         f"({goal_dist:.2f} m) — re-engaging")
                        else:
                            actions[i] = np.zeros(action_dim)        # stay put
                    else:
                        if goal_dist < goal_thresh:
                            reached[i] = True                        # arrived
                            actions[i] = np.zeros(action_dim)
                            logging.info(f"[main] R{i} reached goal")

                # --- Actuation ---
                if controller:
                    # Raw world-frame actions: the policy outputs world-frame
                    # velocities and the controller steers against absolute
                    # angle_z, so no rotation is applied here.
                    controller.execute(actions)
                else:
                    action_str = " ".join(
                        f"R{i}=({actions[i][0]:.2f},{actions[i][1]:.2f})"
                        + ("[OOB]"  if out_of_bounds[i] else
                           "[stop]" if reached[i]       else "")
                        for i in range(len(actions)))
                    logging.info(f"[main] Actions: {action_str}")
            else:
                if controller:
                    controller.stop_all()
                logging.warning("[main] Missing detections — holding position")

            # --- Watchdog ---
            # If too long passes without a good observation (camera stall,
            # detection dropout, slow inference), force-stop so the robots
            # don't keep driving on a latched wheel command with no feedback.
            if time.time() - last_good_obs > obs_timeout and controller:
                controller.stop_all()

            # --- Visualization ---
            if viz:
                actions_viz = (actions
                               if (obs is not None and actions is not None)
                               else np.zeros((n_agents, 2)))
                if viz.update(warped, detections, actions_viz):
                    break  # user pressed 'q'

            # --- Pace the loop ---
            elapsed    = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"[main] Loop overran: {elapsed*1000:.0f} ms "
                                f"(budget {dt*1000:.0f} ms)")

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