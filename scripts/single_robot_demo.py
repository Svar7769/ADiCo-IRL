# single_robot_demo.py — Runs a figure-8 velocity sequence on one Vector robot.
# Shows a live plot of commanded trajectory vs actual odometry.
#
# Usage:
#   python scripts/single_robot_demo.py --demo                      # simulation only
#   python scripts/single_robot_demo.py --demo --serial 00e20117    # real robot

import os
import time
import math
import logging
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import anki_vector
except ImportError:
    anki_vector = None

logger = logging.getLogger(__name__)

# Output directory for saved plots.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "single_robot")


def angle_diff(target: float, current: float) -> float:
    # Returns shortest signed angle difference in [-pi, pi].
    d = (target - current) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def run_demo(serial: Optional[str], duration_s: float = 12.0, hz: float = 10.0):

    # Figure-8 velocity sequence using lemniscate parametric equations.
    dt      = 1.0 / hz
    steps   = int(duration_s * hz)
    t_arr   = np.linspace(0, duration_s, steps)
    u_range = 0.5
    vx_seq  = u_range * np.cos(t_arr * 0.8)        # controls width
    vy_seq  = u_range * np.sin(t_arr * 1.6) * 0.5  # controls height

    # Trajectory storage for plotting.
    cmd_x,  cmd_y  = [0.0], [0.0]  # dead-reckoning — ideal trajectory
    odom_x, odom_y = [], []         # actual robot odometry
    vx_history, vy_history = [], []

    cfg = {
        "wheel_base_mm":         45.0,
        "max_wheel_speed_mm_s":  150.0,
        "turn_wheel_speed_mm_s": 100.0,
        "min_turn_rad":          0.087,
        "u_range":               u_range,
    }

    # Connect to robot if serial provided.
    robot     = None
    connected = False

    if serial:
        if anki_vector is None:
            print("[demo] anki_vector SDK not installed. Running simulation-only mode.")
        else:
            try:
                robot = anki_vector.Robot(serial, behavior_activation_timeout=30)
                robot.connect()
                robot.behavior.set_head_angle(anki_vector.util.degrees(10))
                robot.behavior.set_lift_height(0.0)
                odom_x.append(robot.pose.position.x / 1000.0)
                odom_y.append(robot.pose.position.y / 1000.0)
                connected = True
                print(f"[demo] Connected to Vector {serial}")
            except Exception as e:
                print(f"[demo] Could not connect ({e}). Running simulation-only mode.")
    else:
        print("[demo] No --serial given. Running simulation-only mode.")

    # Matplotlib setup — three plots side by side.
    plt.ion()
    fig, (ax_traj, ax_vx, ax_vy) = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Vector Demo — figure-8 trajectory", fontsize=12)

    ax_traj.set_title("XY Trajectory (m)")
    ax_traj.set_xlabel("x"); ax_traj.set_ylabel("y")
    ax_traj.set_xlim(-0.8, 0.8); ax_traj.set_ylim(-0.8, 0.8)
    ax_traj.set_aspect("equal"); ax_traj.grid(True)
    ax_vx.set_title("Commanded Vx"); ax_vx.set_xlabel("step"); ax_vx.grid(True)
    ax_vy.set_title("Commanded Vy"); ax_vy.set_xlabel("step"); ax_vy.grid(True)

    line_cmd,  = ax_traj.plot([], [], "b-o", ms=3, label="cmd (dead-reckoning)")
    line_odom, = ax_traj.plot([], [], "r-s", ms=3, label="odometry")
    ax_traj.legend(fontsize=7)

    print("[demo] Starting. Press Ctrl-C to stop early.")
    try:
        for step in range(steps):
            t0     = time.time()
            vx, vy = float(vx_seq[step]), float(vy_seq[step])

            # Dead-reckoning — integrate velocity into ideal position.
            cmd_x.append(cmd_x[-1] + vx * dt)
            cmd_y.append(cmd_y[-1] + vy * dt)
            vx_history.append(vx)
            vy_history.append(vy)

            if connected:
                v               = math.hypot(vx, vy)
                theta_target    = math.atan2(vy, vx)
                current_heading = robot.pose.rotation.angle_z.radians
                delta           = angle_diff(theta_target, current_heading)

                # Turn phase — blocking, sleeps through the turn.
                if abs(delta) > cfg["min_turn_rad"]:
                    omega     = 2 * cfg["turn_wheel_speed_mm_s"] / cfg["wheel_base_mm"]
                    turn_time = abs(delta) / omega
                    lw        = -cfg["turn_wheel_speed_mm_s"] if delta > 0 else cfg["turn_wheel_speed_mm_s"]
                    robot.motors.set_wheel_motors(lw, -lw)
                    time.sleep(turn_time)
                    robot.motors.set_wheel_motors(0, 0)
                    time.sleep(0.05)

                # Drive phase — straight for dt seconds, speed clamped to max.
                spd = min(
                    v * cfg["max_wheel_speed_mm_s"] / cfg["u_range"],
                    cfg["max_wheel_speed_mm_s"]
                )
                robot.motors.set_wheel_motors(spd, spd)
                time.sleep(dt)
                robot.motors.set_wheel_motors(0, 0)

                # Read actual position from odometry.
                odom_x.append(robot.pose.position.x / 1000.0)
                odom_y.append(robot.pose.position.y / 1000.0)

            # Update live plots.
            line_cmd.set_data(cmd_x, cmd_y)
            if odom_x:
                line_odom.set_data(odom_x, odom_y)
            ax_vx.clear(); ax_vx.set_title("Commanded Vx"); ax_vx.grid(True)
            ax_vy.clear(); ax_vy.set_title("Commanded Vy"); ax_vy.grid(True)
            ax_vx.plot(vx_history, "b-")
            ax_vy.plot(vy_history, "r-")
            ax_traj.set_title(f"XY Trajectory  step {step+1}/{steps}")
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Sleep remaining time — only needed in simulation mode.
            rem = dt - (time.time() - t0)
            if rem > 0 and not connected:
                time.sleep(rem)

    except KeyboardInterrupt:
        print("\n[demo] Stopped by user.")
    finally:
        if connected:
            robot.motors.set_wheel_motors(0, 0)
            robot.disconnect()

        # Save plot to results/single_robot/
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(RESULTS_DIR, f"trajectory_{timestamp}.png")

        plt.ioff()
        plt.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"[demo] Saved plot → {out}")
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Single Vector robot figure-8 demo")
    p.add_argument("--demo",     action="store_true",   help="Run the demo")
    p.add_argument("--serial",   type=str, default=None, help="Vector serial (e.g. 00e20117)")
    p.add_argument("--duration", type=float, default=12.0, help="Duration in seconds")
    p.add_argument("--hz",       type=float, default=10.0,  help="Control loop rate")
    args = p.parse_args()

    if args.demo:
        run_demo(serial=args.serial, duration_s=args.duration, hz=args.hz)
    else:
        p.print_help()