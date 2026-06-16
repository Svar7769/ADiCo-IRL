# movement_controller.py — Drives real Anki Vector robots using kinematic feedback controller.
# Replaces turn-then-drive with smooth curved motion using differential drive kinematics.

import time
import math
import logging
import concurrent.futures
from typing import List

import anki_vector

logger = logging.getLogger(__name__)


class VectorController:
    # Controls a single Anki Vector robot using kinematic feedback control.
    # Takes (vx, vy) velocity commands and converts to wheel motor speeds.

    def __init__(self, robot: anki_vector.Robot, config: dict):
        self.robot  = robot
        self.config = config

        # Wheel base from config — distance between left and right wheels in mm.
        self.L = config["wheel_base_mm"]

    def execute(self, vx: float, vy: float):
        v = math.hypot(vx, vy)
        if v < 0.01:
            self.robot.motors.set_wheel_motors(0, 0)
            return

        v_mm = v * self.config["max_wheel_speed_mm_s"] / self.config["u_range"]

        theta_target    = math.atan2(vy, vx)
        current_heading = self.robot.pose.rotation.angle_z.radians
        alpha           = self._angle_diff(theta_target, current_heading)

        # Scale forward speed by heading alignment: drive less (or reverse intent)
        # when the target is behind, so the robot turns toward it instead of
        # carving a wide arc at full speed.
        v_mm *= max(0.0, math.cos(alpha))

        omega          = self.config["k_alpha"] * alpha
        turn_component = omega * (self.L / 2)

        # Positive alpha (target CCW of heading) -> turn left -> right wheel faster.
        v_l = v_mm - turn_component
        v_r = v_mm + turn_component

        # Scale both wheels together if either saturates, to preserve the turn ratio.
        max_spd = self.config["max_wheel_speed_mm_s"]
        m = max(abs(v_l), abs(v_r), max_spd)
        scale = max_spd / m
        v_l *= scale
        v_r *= scale

        self.robot.motors.set_wheel_motors(int(v_l), int(v_r))

    def stop(self):
        try:
            self.robot.motors.set_wheel_motors(0, 0)
        except Exception as e:
            logger.error(f"Error stopping robot: {e}")

    def _angle_diff(self, target: float, current: float) -> float:
        # Returns the shortest signed angle difference in [-pi, pi].
        diff = (target - current) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff


class MultiVectorController:
    # Manages multiple VectorController instances, one per robot.
    # Sends commands to all robots in parallel using threads.

    def __init__(self, config: dict):
        self.config      = config
        self.controllers = []

        for agent in config["agents"]:
            serial = agent["serial"]
            if serial == "REPLACE_ME":
                raise ValueError(f"Robot serial for agent {agent['id']} not set")
            robot = self._connect_robot(serial)
            robot.behavior.set_head_angle(anki_vector.util.degrees(10))
            robot.behavior.set_lift_height(0.0)
            self.controllers.append(VectorController(robot, config))

        logger.info(f"Connected to {len(self.controllers)} robots")

    def _connect_robot(self, serial: str) -> anki_vector.Robot:
        # Try connecting up to 3 times with a 5 second delay between attempts.
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
        # Send commands to all robots in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.controllers)) as pool:
            futures = [
                pool.submit(ctrl.execute, *actions[i])
                for i, ctrl in enumerate(self.controllers)
            ]
            concurrent.futures.wait(futures)

    def execute_one(self, index: int, vx: float, vy: float):
        # Send a command to one specific robot by index.
        self.controllers[index].execute(vx, vy)

    def stop_all(self):
        for ctrl in self.controllers:
            ctrl.stop()

    def stop_one(self, index: int):
        self.controllers[index].stop()

    def disconnect(self):
        for ctrl in self.controllers:
            try:
                ctrl.robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")