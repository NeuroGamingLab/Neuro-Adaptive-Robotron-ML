"""
Shared policy for MARL: MLP with move head (9) and fire head (2).
"""
import numpy as np
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .env import RobotronMARLEnv

OBS_DIM = RobotronMARLEnv.OBS_DIM
N_MOVE = RobotronMARLEnv.N_MOVE_ACTIONS
N_FIRE = RobotronMARLEnv.N_FIRE_ACTIONS
HIDDEN = 64
HIDDEN2 = 64


def _layer_init(module: "nn.Module", std: float = np.sqrt(2), bias_const: float = 0.0):
    if not TORCH_AVAILABLE:
        return
    nn.init.orthogonal_(module.weight, std)
    nn.init.constant_(module.bias, bias_const)


if TORCH_AVAILABLE:

    class MARLPolicy(nn.Module):
        def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN, hidden2: int = HIDDEN2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden2),
                nn.Tanh(),
            )
            self.move_head = nn.Linear(hidden2, N_MOVE)
            self.fire_head = nn.Linear(hidden2, N_FIRE)
            self.value_head = nn.Linear(hidden2, 1)
            self.apply(lambda m: _layer_init(m) if isinstance(m, nn.Linear) else None)
            _layer_init(self.move_head, std=0.01)
            _layer_init(self.fire_head, std=0.01)
            _layer_init(self.value_head, std=1.0)

        def get_value(self, obs: torch.Tensor) -> torch.Tensor:
            feats = self.net(obs)
            return self.value_head(feats).squeeze(-1)

        def get_action_and_value(
            self,
            obs: torch.Tensor,
            move_action: torch.Tensor = None,
            fire_action: torch.Tensor = None,
            action_mask: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            feats = self.net(obs)
            move_logits = self.move_head(feats)
            fire_logits = self.fire_head(feats)
            value = self.value_head(feats).squeeze(-1)
            move_probs = torch.softmax(move_logits, dim=-1)
            fire_probs = torch.softmax(fire_logits, dim=-1)
            if action_mask is not None:
                move_probs = move_probs * action_mask
                move_probs = move_probs / (move_probs.sum(-1, keepdim=True) + 1e-8)
            move_dist = torch.distributions.Categorical(probs=move_probs)
            fire_dist = torch.distributions.Categorical(probs=fire_probs)
            if move_action is None:
                move_action = move_dist.sample()
            if fire_action is None:
                fire_action = fire_dist.sample()
            move_entropy = move_dist.entropy()
            fire_entropy = fire_dist.entropy()
            move_log_prob = move_dist.log_prob(move_action)
            fire_log_prob = fire_dist.log_prob(fire_action)
            log_prob = move_log_prob + fire_log_prob
            entropy = move_entropy + fire_entropy
            return move_action, fire_action, log_prob, entropy, value

else:
    MARLPolicy = None  # type: ignore
