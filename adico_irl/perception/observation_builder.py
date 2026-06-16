# observation_builder.py — Builds the 10-dimensional VMAS observation vector per agent.
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List


class ObservationBuilder:
    def __init__(self, config: dict):
        self.vel_ema_alpha = config["vel_ema_alpha"]
        self.max_speed_m_s = config["max_speed_m_s"]
        self.n_agents      = config["n_agents"]
        # Per-agent state — updated every frame.
        self.positions  = [None] * self.n_agents          # last known position (m)
        self.velocities = [np.zeros(2) for _ in range(self.n_agents)]  # EMA velocity
        self.last_times = [None] * self.n_agents           # timestamp of last update

    def update(self, detections: Dict[str, Any]
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[bool]]]:
        # Returns (obs [n_agents,10], measured_vel [n_agents,2], seen [n_agents])
        # or (None, None, None) if any robot has never been seen yet.
        # `seen[i]` is True only if robot i's tag was detected THIS frame;
        # for undetected robots the position is held stale, so the caller
        # should stop them rather than act on the stale value.
        current_time = time.time()
        obs_list = []
        seen = [False] * self.n_agents

        for agent in detections["agents"]:
            i = agent["id"]
            if agent["robot_detected"]:
                seen[i] = True
                new_pos = agent["robot_pos"]
                # Estimate velocity if we have a previous position.
                if self.positions[i] is not None and self.last_times[i] is not None:
                    dt = current_time - self.last_times[i]
                    if dt > 0:
                        raw_vx = (new_pos[0] - self.positions[i][0]) / dt
                        raw_vy = (new_pos[1] - self.positions[i][1]) / dt
                        speed  = np.hypot(raw_vx, raw_vy)
                        if speed > self.max_speed_m_s:
                            # Unrealistic speed — likely a bad detection. Decay.
                            self.velocities[i] = self.velocities[i] * (1 - self.vel_ema_alpha)
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
                    return None, None, None   # never seen — can't build obs yet
                # Decay velocity toward zero while the tag is missing, so a
                # stale obs doesn't keep reporting motion.
                self.velocities[i] = self.velocities[i] * (1 - self.vel_ema_alpha)

            # Build the 10-dim observation vector for this agent.
            x, y   = self.positions[i]
            vx, vy = self.velocities[i]

            # Use goal position if detected, else fall back to robot's own position.
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

            # [x, y, vx, vy, dx_goal, dy_goal, dx_o0, dy_o0, dx_o1, dy_o1]
            obs = [x, y, vx, vy, dx_goal, dy_goal] + dx_others + dy_others
            obs_list.append(obs)

        measured_vel = np.array([self.velocities[a["id"]]
                                 for a in detections["agents"]])
        return np.array(obs_list), measured_vel, seen