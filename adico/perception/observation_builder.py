"""
observation_builder.py — Builds the 10-dimensional VMAS-compatible observation vector per agent.
"""

import numpy as np
import time
from typing import Optional, Dict, Any


class ObservationBuilder:
    """
    Builds the 10-dimensional VMAS-compatible observation vector per agent.

    Attributes:
        vel_ema_alpha (float): EMA alpha for velocity.
        max_speed_m_s (float): Max speed for velocity clamp.
        positions (list): Last known positions.
        velocities (list): EMA velocities.
        last_times (list): Last update times.
    """

    def __init__(self, config: dict):
        """
        Initializes the observation builder.

        Args:
            config (dict): Configuration dictionary.
        """
        self.vel_ema_alpha = config["vel_ema_alpha"]
        self.max_speed_m_s = config["max_speed_m_s"]
        self.n_agents = config["n_agents"]
        self.positions = [None] * self.n_agents
        self.velocities = [np.zeros(2) for _ in range(self.n_agents)]
        self.last_times = [None] * self.n_agents

    def update(self, detections: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Updates internal state and returns obs array or None if any robot missing.

        Args:
            detections (dict): Detection results from ColorDetector.

        Returns:
            Optional[np.ndarray]: Observations (n_agents, obs_dim) or None.
        """
        current_time = time.time()
        obs_list = []

        for agent in detections["agents"]:
            i = agent["id"]
            if agent["robot_detected"]:
                new_pos = agent["robot_pos"]
                if self.positions[i] is not None and self.last_times[i] is not None:
                    dt = current_time - self.last_times[i]
                    if dt > 0:
                        raw_vx = (new_pos[0] - self.positions[i][0]) / dt
                        raw_vy = (new_pos[1] - self.positions[i][1]) / dt
                        speed = np.hypot(raw_vx, raw_vy)
                        if speed > self.max_speed_m_s:
                            # Decay velocity
                            decay_factor = (1 - self.vel_ema_alpha)
                            self.velocities[i] *= decay_factor
                        else:
                            self.velocities[i] = (
                                self.vel_ema_alpha * np.array([raw_vx, raw_vy]) +
                                (1 - self.vel_ema_alpha) * self.velocities[i]
                            )
                self.positions[i] = new_pos
                self.last_times[i] = current_time
            else:
                # Hold position, no velocity update
                if self.positions[i] is None:
                    return None  # Missing initial detection

            # Build obs for this agent
            x, y = self.positions[i]
            vx, vy = self.velocities[i]
            gx, gy = agent["goal_pos"] if agent["goal_detected"] else self.positions[i]  # Fallback to self if goal missing
            dx_goal = x - gx
            dy_goal = y - gy

            # Other agents
            dx_others = []
            dy_others = []
            for j in range(self.n_agents):
                if j != i:
                    if self.positions[j] is not None:
                        dx_others.append(x - self.positions[j][0])
                        dy_others.append(y - self.positions[j][1])
                    else:
                        dx_others.append(0.0)
                        dy_others.append(0.0)

            obs = [x, y, vx, vy, dx_goal, dy_goal] + dx_others + dy_others
            obs_list.append(obs)

        return np.array(obs_list)