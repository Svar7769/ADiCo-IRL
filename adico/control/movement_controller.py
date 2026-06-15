"""
movement_controller.py — Drives real Anki Vector robots using turn-then-drive approach.
"""

import time
import math
import logging
import concurrent.futures
from typing import List, Optional

import anki_vector

logger = logging.getLogger(__name__)


class MultiVectorController:
    """
    Drives real Anki Vector robots using turn-then-drive approach.

    Attributes:
        robots (List[anki_vector.Robot]): List of connected robots.
        config (dict): Configuration dictionary.
    """

    def __init__(self, config: dict):
        """
        Initializes the controller and connects to robots.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.robots = []
        for agent in config["agents"]:
            serial = agent["serial"]
            if serial == "REPLACE_ME":
                raise ValueError(f"Robot serial for agent {agent['id']} not set")
            robot = self._connect_robot(serial)
            robot.behavior.set_head_angle(anki_vector.util.degrees(10))
            robot.behavior.set_lift_height(0.0)
            self.robots.append(robot)
        logger.info(f"Connected to {len(self.robots)} robots")

    def _connect_robot(self, serial: str) -> anki_vector.Robot:
        """
        Connects to a robot with retry logic.

        Args:
            serial (str): Robot serial number.

        Returns:
            anki_vector.Robot: Connected robot.
        """
        for attempt in range(3):
            try:
                robot = anki_vector.Robot(serial, behavior_activation_timeout=30)
                robot.connect()
                return robot
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1} failed for {serial}: {e}")
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError(f"Failed to connect to robot {serial}")

    def execute(self, actions: List[List[float]]):
        """
        Executes actions for all robots in parallel.

        Args:
            actions (List[List[float]]): Actions [[vx0, vy0], [vx1, vy1], ...]
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.robots)) as pool:
            futures = [
                pool.submit(self._execute_single, robot, *actions[i])
                for i, robot in enumerate(self.robots)
            ]
            concurrent.futures.wait(futures)

    def _execute_single(self, robot: anki_vector.Robot, vx: float, vy: float):
        """
        Issues a single non-blocking motor command. The loop rate acts as the timer.

        Args:
            robot (anki_vector.Robot): The robot.
            vx (float): X velocity.
            vy (float): Y velocity.
        """
        v = math.hypot(vx, vy)
        if v < 0.01:
            robot.motors.set_wheel_motors(0, 0)
            return

        theta_target = math.atan2(vy, vx)
        current_heading = robot.pose.rotation.angle_z.radians
        delta = self._angle_diff(theta_target, current_heading)

        if abs(delta) > self.config["min_turn_rad"]:
            # Turn in place — command and return; next loop tick re-evaluates
            turn_speed = self.config["turn_wheel_speed_mm_s"]
            sign = 1.0 if delta > 0 else -1.0
            robot.motors.set_wheel_motors(-sign * turn_speed, sign * turn_speed)
        else:
            # Heading aligned — drive straight
            speed = min(
                v * self.config["max_wheel_speed_mm_s"] / self.config["u_range"],
                self.config["max_wheel_speed_mm_s"]
            )
            robot.motors.set_wheel_motors(speed, speed)

    def _angle_diff(self, target: float, current: float) -> float:
        """
        Computes shortest signed angle difference.

        Args:
            target (float): Target angle.
            current (float): Current angle.

        Returns:
            float: Signed difference in [-pi, pi].
        """
        diff = (target - current) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def stop_all(self):
        """
        Stops all robots.
        """
        for robot in self.robots:
            try:
                robot.motors.set_wheel_motors(0, 0)
            except Exception as e:
                logger.error(f"Error stopping robot: {e}")

    def disconnect(self):
        """
        Disconnects from all robots.
        """
        for robot in self.robots:
            try:
                robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting robot: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo: python movement_controller.py --demo [--serial XXXX]
# ──────────────────────────────────────────────────────────────────────────────

def _run_demo(serial: Optional[str], duration_s: float = 12.0, hz: float = 10.0):
    """
    Runs a scripted figure-8 velocity sequence on one Vector robot and shows
    a live matplotlib plot of:
      - Commanded (vx, vy) trajectory (integrated dead-reckoning)
      - Actual odometry position reported by the robot
    Press Ctrl-C to stop early.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # Scripted (vx, vy) sequence — figure-8-ish pattern
    dt = 1.0 / hz
    steps = int(duration_s * hz)
    t_arr = np.linspace(0, duration_s, steps)
    u_range = 0.5
    # lemniscate parametric: vx = u*cos(t), vy = u*sin(2t)/2
    vx_seq = u_range * np.cos(t_arr * 0.8)
    vy_seq = u_range * np.sin(t_arr * 1.6) * 0.5

    # Storage for plotting
    cmd_x, cmd_y   = [0.0], [0.0]   # dead-reckoning integral of commands
    odom_x, odom_y = [], []          # actual robot odometry (if connected)

    robot = None
    connected = False
    if serial:
        try:
            robot = anki_vector.Robot(serial, behavior_activation_timeout=30)
            robot.connect()
            robot.behavior.set_head_angle(anki_vector.util.degrees(10))
            robot.behavior.set_lift_height(0.0)
            odom_x.append(robot.pose.position.x / 1000.0)  # mm → m
            odom_y.append(robot.pose.position.y / 1000.0)
            connected = True
            print(f"[demo] Connected to Vector {serial}")
        except Exception as e:
            print(f"[demo] Could not connect ({e}). Running in simulation-only mode.")
    else:
        print("[demo] No --serial given. Running simulation-only mode.")

    # ── matplotlib setup ──
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Vector Demo — figure-8 trajectory", fontsize=12)

    ax_traj, ax_vx, ax_vy = axes
    ax_traj.set_title("XY Trajectory (m)")
    ax_traj.set_xlabel("x"); ax_traj.set_ylabel("y")
    ax_traj.set_xlim(-0.8, 0.8); ax_traj.set_ylim(-0.8, 0.8)
    ax_traj.set_aspect("equal"); ax_traj.grid(True)

    ax_vx.set_title("Commanded Vx"); ax_vx.set_xlabel("step"); ax_vx.set_ylabel("m/s"); ax_vx.grid(True)
    ax_vy.set_title("Commanded Vy"); ax_vy.set_xlabel("step"); ax_vy.set_ylabel("m/s"); ax_vy.grid(True)

    line_cmd,  = ax_traj.plot([], [], "b-o", ms=3, label="cmd (dead-reckoning)")
    line_odom, = ax_traj.plot([], [], "r-s", ms=3, label="odometry")
    ax_traj.legend(fontsize=7)

    vx_history, vy_history = [], []

    # ── control loop ──
    controller_cfg = {
        "wheel_base_mm": 45.0,
        "max_wheel_speed_mm_s": 150.0,
        "turn_wheel_speed_mm_s": 100.0,
        "min_turn_rad": 0.087,
        "control_hz": hz,
        "u_range": u_range,
    }

    def _angle_diff(target, current):
        d = (target - current) % (2 * math.pi)
        if d > math.pi:
            d -= 2 * math.pi
        return d

    print("[demo] Starting. Press Ctrl-C to stop early.")
    try:
        for step in range(steps):
            t0 = time.time()
            vx, vy = float(vx_seq[step]), float(vy_seq[step])

            # ── dead-reckoning integration ──
            cx = cmd_x[-1] + vx * dt
            cy = cmd_y[-1] + vy * dt
            cmd_x.append(cx); cmd_y.append(cy)
            vx_history.append(vx); vy_history.append(vy)

            # ── send to robot ──
            if connected:
                v = math.hypot(vx, vy)
                theta_target = math.atan2(vy, vx)
                current_heading = robot.pose.rotation.angle_z.radians
                delta = _angle_diff(theta_target, current_heading)

                if abs(delta) > controller_cfg["min_turn_rad"]:
                    omega = 2 * controller_cfg["turn_wheel_speed_mm_s"] / controller_cfg["wheel_base_mm"]
                    turn_time = abs(delta) / omega
                    lw = -controller_cfg["turn_wheel_speed_mm_s"] if delta > 0 else controller_cfg["turn_wheel_speed_mm_s"]
                    rw = -lw
                    robot.motors.set_wheel_motors(lw, rw)
                    time.sleep(turn_time)
                    robot.motors.set_wheel_motors(0, 0)
                    time.sleep(0.05)

                spd = v * controller_cfg["max_wheel_speed_mm_s"] / controller_cfg["u_range"]
                robot.motors.set_wheel_motors(spd, spd)
                time.sleep(dt)
                robot.motors.set_wheel_motors(0, 0)

                px = robot.pose.position.x / 1000.0
                py = robot.pose.position.y / 1000.0
                odom_x.append(px); odom_y.append(py)

            # ── update plot ──
            line_cmd.set_data(cmd_x, cmd_y)
            if odom_x:
                line_odom.set_data(odom_x, odom_y)
            ax_vx.clear(); ax_vx.set_title("Commanded Vx"); ax_vx.grid(True)
            ax_vy.clear(); ax_vy.set_title("Commanded Vy"); ax_vy.grid(True)
            ax_vx.plot(vx_history, "b-"); ax_vy.plot(vy_history, "r-")
            ax_traj.set_title(f"XY Trajectory  step {step+1}/{steps}")
            fig.canvas.draw(); fig.canvas.flush_events()

            elapsed = time.time() - t0
            rem = dt - elapsed
            if rem > 0 and not connected:
                time.sleep(rem)

    except KeyboardInterrupt:
        print("\n[demo] Stopped by user.")
    finally:
        if connected:
            robot.motors.set_wheel_motors(0, 0)
            robot.disconnect()
        plt.ioff()
        plt.tight_layout()
        out = "demo_trajectory.png"
        fig.savefig(out, dpi=120)
        print(f"[demo] Saved plot → {out}")
        plt.show()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Movement controller standalone demo")
    p.add_argument("--demo",   action="store_true", help="Run figure-8 demo")
    p.add_argument("--serial", type=str,  default=None, help="Vector serial (e.g. 00e20117)")
    p.add_argument("--duration", type=float, default=12.0, help="Demo duration in seconds")
    p.add_argument("--hz",    type=float, default=10.0,  help="Control loop rate")
    args = p.parse_args()

    if args.demo:
        _run_demo(serial=args.serial, duration_s=args.duration, hz=args.hz)
    else:
        p.print_help()