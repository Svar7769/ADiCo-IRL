# observation_builder.py — Builds the 10-dimensional VMAS observation vector per agent.

import time
import numpy as np
from typing import Optional, Dict, Any


class ObservationBuilder:

    def __init__(self, config: dict):
        self.vel_ema_alpha = config["vel_ema_alpha"]
        self.max_speed_m_s = config["max_speed_m_s"]
        self.n_agents      = config["n_agents"]

        # Per-agent state — updated every frame.
        self.positions  = [None] * self.n_agents          # last known position in meters
        self.velocities = [np.zeros(2)] * self.n_agents   # EMA smoothed velocity
        self.last_times = [None] * self.n_agents           # timestamp of last update

    def update(self, detections: Dict[str, Any]) -> Optional[np.ndarray]:
        # Returns obs array [n_agents, 10] or None if any robot is missing on first frame.
        current_time = time.time()
        obs_list = []

        for agent in detections["agents"]:
            i = agent["id"]

            if agent["robot_detected"]:
                new_pos = agent["robot_pos"]

                # Estimate velocity if we have a previous position.
                if self.positions[i] is not None and self.last_times[i] is not None:
                    dt = current_time - self.last_times[i]
                    if dt > 0:
                        raw_vx = (new_pos[0] - self.positions[i][0]) / dt
                        raw_vy = (new_pos[1] - self.positions[i][1]) / dt
                        speed  = np.hypot(raw_vx, raw_vy)

                        if speed > self.max_speed_m_s:
                            # Unrealistic speed — likely a bad detection. Decay toward zero.
                            self.velocities[i] *= (1 - self.vel_ema_alpha)
                        else:
                            # EMA smooth the velocity.
                            self.velocities[i] = (
                                self.vel_ema_alpha * np.array([raw_vx, raw_vy]) +
                                (1 - self.vel_ema_alpha) * self.velocities[i]
                            )

                self.positions[i]  = new_pos
                self.last_times[i] = current_time

            else:
                # Robot not detected this frame — hold last known position.
                if self.positions[i] is None:
                    return None  # Never seen this robot — can't build obs yet.

            # Build the 10-dim observation vector for this agent.
            x, y   = self.positions[i]
            vx, vy = self.velocities[i]

            # Use goal position if detected, otherwise fall back to robot's own position.
            gx, gy = agent["goal_pos"] if agent["goal_detected"] else self.positions[i]
            dx_goal = x - gx
            dy_goal = y - gy

            # Relative positions to all other agents.
            dx_others, dy_others = [], []
            for j in range(self.n_agents):
                if j != i:
                    if self.positions[j] is not None:
                        dx_others.append(x - self.positions[j][0])
                        dy_others.append(y - self.positions[j][1])
                    else:
                        dx_others.append(0.0)
                        dy_others.append(0.0)

            # [x, y, vx, vy, dx_goal, dy_goal, dx_other0, dy_other0, dx_other1, dy_other1]
            obs = [x, y, vx, vy, dx_goal, dy_goal] + dx_others + dy_others
            obs_list.append(obs)

        return np.array(obs_list)