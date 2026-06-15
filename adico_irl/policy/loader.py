# loader.py — Loads a trained ADiCo policy from a BenchMARL checkpoint .pt file.
#
# Checkpoint format (navigation_3a3g_seed0.pt):
#   Actor weights live under:
#     collector.policy_state_dict.module.0.module.0.module.0.{shared_mlp, agent_mlps}
#
# Forward pass:
#   shared_loc  = shared_mlp(obs_i)[:action_dim]
#   agent_loc   = shared_loc + agent_mlps[i](obs_i)
#   action_i    = tanh(agent_loc) * u_range

import logging
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Prefix path to actor weights inside the checkpoint.
ACTOR_PREFIX = (
    "collector.policy_state_dict."
    "module.0.module.0.module.0."
)


# ── Network building blocks ────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    # Shared MLP — same weights run for every agent.
    # obs_dim → [Linear → Tanh] × n_layers → action_dim * 2 (loc + scale)
    # Only loc (first half) is used at inference.

    def __init__(self, obs_dim: int, num_cells: List[int], out_dim: int):
        super().__init__()
        layers = []
        in_d = obs_dim
        for hidden in num_cells:
            layers += [nn.Linear(in_d, hidden), nn.Tanh()]
            in_d = hidden
        layers.append(nn.Linear(in_d, out_dim))
        self.params = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.params:
            x = layer(x)
        return x


class BatchedLinear(nn.Module):
    # One linear layer with separate weights per agent stored in a single stacked tensor.
    # weight: [n_agents, out_features, in_features]
    # bias:   [n_agents, out_features]

    def __init__(self, n_agents: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_agents, out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(n_agents, out_features))

    def forward_agent(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return x @ self.weight[i].T + self.bias[i]


class AgentMLPs(nn.Module):
    # Per-agent MLP using BatchedLinear — each agent has its own private weights.
    # Outputs an offset [action_dim] added on top of the shared MLP output.

    def __init__(self, obs_dim: int, num_cells: List[int], out_dim: int, n_agents: int):
        super().__init__()
        layers = []
        in_d = obs_dim
        for hidden in num_cells:
            layers += [BatchedLinear(n_agents, in_d, hidden), nn.Tanh()]
            in_d = hidden
        layers.append(BatchedLinear(n_agents, in_d, out_dim))
        self.params = nn.ModuleList(layers)

    def forward_agent(self, x: torch.Tensor, i: int) -> torch.Tensor:
        for layer in self.params:
            if isinstance(layer, BatchedLinear):
                x = layer.forward_agent(x, i)
            else:
                x = torch.tanh(x)
        return x


class DiCoPolicy(nn.Module):
    # Full ADiCo actor: shared backbone + per-agent head.
    # Input:  obs [n_agents, obs_dim]
    # Output: actions [n_agents, action_dim]  bounded to [-u_range, u_range]

    def __init__(self, obs_dim: int, n_agents: int, num_cells: List[int], action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.shared_mlp = SharedMLP(obs_dim, num_cells, action_dim * 2)
        self.agent_mlps = AgentMLPs(obs_dim, num_cells, action_dim, n_agents)

    def forward(self, obs: torch.Tensor, u_range: float = 0.5) -> torch.Tensor:
        actions = []
        for i, obs_i in enumerate(obs):
            shared_loc = self.shared_mlp(obs_i)[: self.action_dim]
            agent_out  = self.agent_mlps.forward_agent(obs_i, i)
            action     = torch.tanh(shared_loc + agent_out) * u_range
            actions.append(action)
        return torch.stack(actions)


# Checkpoint loading 

def load_policy(
    checkpoint_path: str,
    obs_dim: int,
    n_agents: int,
    num_cells: List[int] = None,
    action_dim: int = 2,
    u_range: float = 0.5,
    device: str = "cpu",
) -> Callable:

    num_cells = num_cells or [256, 256]
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {ckpt_path.name}  (obs_dim={obs_dim}, n_agents={n_agents})")

    # Load and flatten the nested checkpoint dict.
    raw  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    flat = _flatten(raw)

    # Strip the long BenchMARL prefix — keep only actor weight tensors.
    actor_weights = {
        k[len(ACTOR_PREFIX):]: v
        for k, v in flat.items()
        if k.startswith(ACTOR_PREFIX) and isinstance(v, torch.Tensor)
    }

    if not actor_weights:
        raise RuntimeError(
            f"No weights found under prefix '{ACTOR_PREFIX}'.\n"
            f"Top-level keys (first 10): {list(flat.keys())[:10]}"
        )

    logger.info(f"Extracted {len(actor_weights)} actor tensors.")

    # Build a fresh model and load the checkpoint weights into it.
    policy   = DiCoPolicy(obs_dim, n_agents, num_cells, action_dim)
    expected = set(policy.state_dict().keys())
    weights  = {k: v for k, v in actor_weights.items() if k in expected}
    missing  = expected - set(weights.keys())

    if missing:
        raise RuntimeError(f"Missing expected policy weights: {missing}")

    policy.load_state_dict(weights, strict=True)
    logger.info(f"Policy loaded — {sum(p.numel() for p in policy.parameters()):,} parameters.")

    policy   = policy.to(device).eval()
    _u_range = u_range

    # Return a clean inference function — no gradients, no PyTorch boilerplate for the caller.
    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return policy(obs.to(device), _u_range).detach()

    return policy_fn


def _flatten(obj, prefix: str = "") -> dict:
    # Recursively collapse a nested dict into a flat dict with dot-joined keys.
    result = {}
    if isinstance(obj, dict) or hasattr(obj, "items"):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten(v, full))
    else:
        result[prefix] = obj
    return result