# single_robot_demo.py — Runs a figure-8 velocity sequence on one Vector robot.
# Shows a live plot of commanded trajectory vs actual odometry.
#
# Usage:
#   python scripts/single_robot_demo.py --demo                   # simulation only
#   python scripts/single_robot_demo.py --demo --serial 00506502 # real robot

import os
import sys
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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from adico_irl.control.movement_controller import VectorController

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "single_robot")

CFG = {
    "wheel_base_mm":        48.0,
    "max_wheel_speed_mm_s": 150.0,
    "u_range":              0.4,
    "k_alpha":              6.0,
}

EMA_ALPHA    = 0.3     # smoothing for the measured-velocity estimate
RENDER_EVERY = 2       # redraw every N steps to keep the control loop fast


def make_figure_eight(steps):
    # Lissajous figure-8: pos (A sin t, B sin 2t) -> vel (A cos t, 2B cos 2t).
    t = np.linspace(0, 2 * math.pi, steps, endpoint=False)
    A, B = 0.10, 0.05
    return A * np.cos(t), 2 * B * np.cos(2 * t)


def connect_robot(serial):
    # Returns (robot, controller, start_pose) or None if connection fails.
    if anki_vector is None:
        print("[demo] anki_vector SDK not installed — simulation only.")
        return None
    try:
        robot = anki_vector.Robot(serial, behavior_activation_timeout=30)
        robot.connect()
        robot.behavior.set_head_angle(anki_vector.util.degrees(10))
        robot.behavior.set_lift_height(0.0)
        time.sleep(0.5)
        controller = VectorController(robot, CFG)
        start = (robot.pose.position.x / 1000.0,
                 robot.pose.position.y / 1000.0,
                 robot.pose.rotation.angle_z.radians)
        print(f"[demo] Connected to Vector {serial}")
        print(f"[demo] Start pose: ({start[0]:.3f}, {start[1]:.3f}) m, "
              f"heading {math.degrees(start[2]):.1f} deg")
        return robot, controller, start
    except Exception as e:
        print(f"[demo] Could not connect ({e}) — simulation only.")
        return None


def setup_plot():
    plt.ion()
    fig, ((ax_traj, ax_vx), (ax_vy, ax_err)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Vector Demo — figure-8 trajectory", fontsize=12)

    ax_traj.set(title="XY Trajectory (m)", xlabel="x", ylabel="y",
                xlim=(-1, 1), ylim=(-1, 1))
    ax_traj.set_aspect("equal"); ax_traj.grid(True)

    for ax, title in ((ax_vx, "Vx — cmd vs actual"),
                      (ax_vy, "Vy — cmd vs actual"),
                      (ax_err, "Error")):
        ax.set(title=title, xlabel="step")
        ax.grid(True)
    ax_vx.set_ylabel("m/s"); ax_vy.set_ylabel("m/s"); ax_err.set_ylabel("m / m/s")
    ax_err.axhline(0, color="gray", linewidth=0.5)

    # Pre-created line handles, updated in-loop via set_data (no clear/replot).
    lines = {
        "cmd":    ax_traj.plot([], [], "b-o", ms=2, label="cmd (dead-reckoning)")[0],
        "odom":   ax_traj.plot([], [], "r-s", ms=2, label="odometry")[0],
        "vx_cmd": ax_vx.plot([], [], "b-", label="commanded")[0],
        "vx_act": ax_vx.plot([], [], "r-", label="actual")[0],
        "vy_cmd": ax_vy.plot([], [], "b-", label="commanded")[0],
        "vy_act": ax_vy.plot([], [], "r-", label="actual")[0],
        "pos_e":  ax_err.plot([], [], "g-",  label="position error (m)")[0],
        "vx_e":   ax_err.plot([], [], "b--", alpha=0.6, label="vx error (m/s)")[0],
        "vy_e":   ax_err.plot([], [], "r--", alpha=0.6, label="vy error (m/s)")[0],
    }
    for ax in (ax_traj, ax_vx, ax_vy, ax_err):
        ax.legend(fontsize=7)
    return fig, (ax_vx, ax_vy, ax_err), lines


def run_demo(serial: Optional[str], duration_s: float = 100.0, hz: float = 10.0):
    dt    = 1.0 / hz
    steps = int(duration_s * hz)
    vx_seq, vy_seq = make_figure_eight(steps)

    conn = connect_robot(serial) if serial else None
    if conn:
        robot, controller, (start_x, start_y, start_th) = conn
    else:
        if serial is None:
            print("[demo] No --serial given — simulation only.")
        robot = controller = None

    # Logged series.
    cmd_x,  cmd_y  = [0.0], [0.0]
    odom_x, odom_y = [], []
    vx_cmd, vy_cmd = [], []
    vx_act, vy_act = [], []
    pos_err_log    = []
    vx_err, vy_err = [], []

    vx_ema = vy_ema = 0.0
    sim_x  = sim_y  = 0.0
    max_v  = CFG["max_wheel_speed_mm_s"] / 1000.0

    fig, (ax_vx, ax_vy, ax_err), lines = setup_plot()

    print("[demo] Starting. Press Ctrl-C to stop early.")
    try:
        for step in range(steps):
            t0 = time.time()
            vx, vy = float(vx_seq[step]), float(vy_seq[step])

            # Ideal dead-reckoned path.
            cmd_x.append(cmd_x[-1] + vx * dt)
            cmd_y.append(cmd_y[-1] + vy * dt)
            vx_cmd.append(vx); vy_cmd.append(vy)

            if controller:
                # Rotate command into the robot's world frame so command and
                # odometry share one frame regardless of boot heading.
                controller.execute(vx, vy)
                time.sleep(dt)
                real_dt = time.time() - t0

                # Raw odometry, origin-shifted to start (no derotation needed).
                ox = robot.pose.position.x / 1000.0 - start_x
                oy = robot.pose.position.y / 1000.0 - start_y
                odom_x.append(ox); odom_y.append(oy)
                pos_err_log.append(math.hypot(cmd_x[-1] - ox, cmd_y[-1] - oy))

                WINDOW = 3
                if len(odom_x) > WINDOW:
                    # Calculate the average velocity over the last WINDOW steps.
                    span_dt = WINDOW * dt
                    dx = odom_x[-1] - odom_x[-1 - WINDOW]
                    dy = odom_y[-1] - odom_y[-1 - WINDOW]
                    rvx = max(-max_v, min(max_v, dx / span_dt))
                    rvy = max(-max_v, min(max_v, dy / span_dt))
                    vx_ema = EMA_ALPHA * rvx + (1 - EMA_ALPHA) * vx_ema
                    vy_ema = EMA_ALPHA * rvy + (1 - EMA_ALPHA) * vy_ema
                    vx_act.append(vx_ema); vy_act.append(vy_ema)
                    vx_err.append(vx - vx_ema); vy_err.append(vy - vy_ema)
            else:
                # Sim-only: integrate command as synthetic "actual" odometry.
                sim_x += vx * dt; sim_y += vy * dt
                odom_x.append(sim_x); odom_y.append(sim_y)
                pos_err_log.append(0.0)
                vx_act.append(vx); vy_act.append(vy)
                vx_err.append(0.0); vy_err.append(0.0)

            # Redraw (throttled; last step always drawn).
            if step % RENDER_EVERY == 0 or step == steps - 1:
                lines["cmd"].set_data(cmd_x, cmd_y)
                lines["odom"].set_data(odom_x, odom_y)
                xs = range(len(vx_cmd))
                lines["vx_cmd"].set_data(xs, vx_cmd)
                lines["vy_cmd"].set_data(xs, vy_cmd)
                xa = range(len(vx_act))
                lines["vx_act"].set_data(xa, vx_act)
                lines["vy_act"].set_data(xa, vy_act)
                lines["pos_e"].set_data(range(len(pos_err_log)), pos_err_log)
                lines["vx_e"].set_data(range(len(vx_err)), vx_err)
                lines["vy_e"].set_data(range(len(vy_err)), vy_err)
                for ax in (ax_vx, ax_vy, ax_err):
                    ax.relim(); ax.autoscale_view()
                fig.suptitle(f"Vector Demo — figure-8  step {step+1}/{steps}",
                             fontsize=12)
                fig.canvas.draw(); fig.canvas.flush_events()

            # Pace the sim loop (connected loop already slept dt above).
            rem = dt - (time.time() - t0)
            if rem > 0 and not controller:
                time.sleep(rem)

    except KeyboardInterrupt:
        print("\n[demo] Stopped by user.")
    finally:
        if controller:
            controller.stop()
            robot.disconnect()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.abspath(os.path.join(
            RESULTS_DIR, f"trajectory_{time.strftime('%Y%m%d_%H%M%S')}.png"))
        plt.ioff(); plt.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"[demo] Saved plot → {out}")
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Single Vector robot figure-8 demo")
    p.add_argument("--demo",     action="store_true",    help="Run the demo")
    p.add_argument("--serial",   type=str,   default=None, help="Vector serial")
    p.add_argument("--duration", type=float, default=12.0, help="Duration (s)")
    p.add_argument("--hz",       type=float, default=10.0,  help="Loop rate (Hz)")
    args = p.parse_args()

    run_demo(args.serial, args.duration, args.hz) if args.demo else p.print_help()