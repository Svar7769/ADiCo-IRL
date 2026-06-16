# test_controller.py — tests VectorController on a real robot.
#
# Usage:
#   python scripts/test_controller.py --serial 00a12dbd 

import math
import sys
import os
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from adico_irl.control.movement_controller import VectorController

import anki_vector

cfg = {
    "wheel_base_mm":        45.0,
    "max_wheel_speed_mm_s": 150.0,
    "u_range":              0.4,
    "k_alpha":              3.0,
}

def run_test(serial: str):
    print(f"[test] Connecting to {serial}...")
    robot = anki_vector.Robot(serial, behavior_activation_timeout=30)
    robot.connect()
    robot.behavior.set_head_angle(anki_vector.util.degrees(10))
    robot.behavior.set_lift_height(0.0)
    time.sleep(0.5)
    print(f"[test] Connected. Starting heading: {math.degrees(robot.pose.rotation.angle_z.radians):.1f}°")

    ctrl = VectorController(robot, cfg)

    try:
        print("\n--- Test 1: heading aligned (vx=0.4, vy=0.0) ---")
        print("  Expected: drive straight forward at full speed")
        ctrl.execute(0.4, 0.0)
        time.sleep(2.0)
        ctrl.stop()
        time.sleep(0.5)

        print("\n--- Test 2: pure turn (vx=0.0, vy=0.4) ---")
        print("  Expected: curve left")
        ctrl.execute(0.0, 0.4)
        time.sleep(2.0)
        ctrl.stop()
        time.sleep(0.5)

        print("\n--- Test 3: diagonal (vx=0.4, vy=0.4) ---")
        print("  Expected: curve forward-left")
        ctrl.execute(0.4, 0.4)
        time.sleep(2.0)
        ctrl.stop()
        time.sleep(0.5)

        print("\n--- Test 4: near zero action ---")
        print("  Expected: stop immediately")
        ctrl.execute(0.001, 0.0)
        time.sleep(1.0)
        ctrl.stop()

        print("\n[test] All tests done.")

    except KeyboardInterrupt:
        print("\n[test] Interrupted.")
    finally:
        ctrl.stop()
        robot.disconnect()
        print("[test] Disconnected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VectorController real robot test")
    p.add_argument("--serial", type=str, required=True, help="Robot serial e.g. 00e20117")
    args = p.parse_args()
    run_test(args.serial)