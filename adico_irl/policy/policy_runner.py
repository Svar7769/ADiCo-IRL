# runner.py — Thin wrapper around loader.py for use in the main loop.
# Handles numpy ↔ torch conversion so the rest of the pipeline never touches PyTorch.

import numpy as np
import torch
from adico_irl.policy.loader import load_policy


class PolicyRunner:

    def __init__(self, config: dict):
        # Load the trained policy once at startup.
        self.policy_fn = load_policy(
            checkpoint_path = config["checkpoint_path"],
            obs_dim         = config["obs_dim"],
            n_agents        = config["n_agents"],
            action_dim      = config["action_dim"],
            u_range         = config["u_range"],
        )

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        # obs [n_agents, obs_dim] numpy → actions [n_agents, action_dim] numpy
        obs_tensor = torch.from_numpy(obs).float()
        return self.policy_fn(obs_tensor).numpy()