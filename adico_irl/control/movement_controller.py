# movement_controller.py — Drives real Anki Vector robots using turn-then-drive approach.

# We need to improve this so the controller is for one agent and then it can we used for multiple robot through a warapper. 

import time
import math
import logging
import concurrent.futures
from typing import List

import anki_vector

logger = logging.getLogger(__name__)


class MultiVectorController:

    def __init__(self, config: dict):
        self.config = config
        self.robots = []

        for agent in config["agents"]:
            serial = agent["serial"]
            if serial == "REPLACE_ME":
                raise ValueError(f"Robot serial for agent {agent['id']} not set")
            robot = self._connect_robot(serial)
            # Set head and lift to neutral position.
            robot.behavior.set_head_angle(anki_vector.util.degrees(10))
            robot.behavior.set_lift_height(0.0)
            self.robots.append(robot)

        logger.info(f"Connected to {len(self.robots)} robots")

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
        # Send commands to all robots in parallel using threads.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.robots)) as pool:
            futures = [
                pool.submit(self._execute_single, robot, *actions[i])
                for i, robot in enumerate(self.robots)
            ]
            concurrent.futures.wait(futures)

    def _execute_single(self, robot: anki_vector.Robot, vx: float, vy: float):
        # Non-blocking motor command — loop rate acts as the timer.
        v = math.hypot(vx, vy)

        # Stop if action magnitude is near zero.
        if v < 0.01:
            robot.motors.set_wheel_motors(0, 0)
            return

        theta_target    = math.atan2(vy, vx)
        current_heading = robot.pose.rotation.angle_z.radians
        delta           = self._angle_diff(theta_target, current_heading)

        if abs(delta) > self.config["min_turn_rad"]:
            # Heading error too large — turn in place, re-evaluate next tick.
            turn_speed = self.config["turn_wheel_speed_mm_s"]
            sign       = 1.0 if delta > 0 else -1.0
            robot.motors.set_wheel_motors(-sign * turn_speed, sign * turn_speed)
        else:
            # Heading aligned — drive straight.
            speed = min(
                v * self.config["max_wheel_speed_mm_s"] / self.config["u_range"],
                self.config["max_wheel_speed_mm_s"]
            )
            robot.motors.set_wheel_motors(speed, speed)

    def _angle_diff(self, target: float, current: float) -> float:
        # Returns the shortest signed angle difference in [-pi, pi].
        diff = (target - current) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def stop_all(self):
        # Stop all robots. Each robot wrapped independently so one failure doesn't block others.
        for robot in self.robots:
            try:
                robot.motors.set_wheel_motors(0, 0)
            except Exception as e:
                logger.error(f"Error stopping robot: {e}")

    def stop_one(self, index: int):
        # Stop a single robot by index.
        try:
            self.robots[index].motors.set_wheel_motors(0, 0)
        except Exception as e:
            logger.error(f"Error stopping robot {index}: {e}")

    def disconnect(self):
        # Disconnect from all robots cleanly.
        for robot in self.robots:
            try:
                robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting robot: {e}")