"""
policy_runner.py — Wrapper around policy/loader.py for use in the main loop.
"""

import numpy as np
import torch
from typing import Callable

from adico.policy.loader import load_policy


class PolicyRunner:
    """
    Wrapper around policy/loader.py for use in the main loop.

    Attributes:
        policy_fn (Callable): The loaded policy function.
    """

    def __init__(self, config: dict):
        """
        Initializes the policy runner.

        Args:
            config (dict): Configuration dictionary.
        """
        self.policy_fn = load_policy(
            checkpoint_path=config["checkpoint_path"],
            obs_dim=config["obs_dim"],
            n_agents=config["n_agents"],
            action_dim=config["action_dim"],
            u_range=config["u_range"]
        )

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Converts obs to torch tensor, calls policy, returns numpy actions.

        Args:
            obs (np.ndarray): Observations (n_agents, obs_dim).

        Returns:
            np.ndarray: Actions (n_agents, action_dim).
        """
        obs_tensor = torch.from_numpy(obs).float()
        actions_tensor = self.policy_fn(obs_tensor)
        return actions_tensor.numpy()