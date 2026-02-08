"""
Simple DNN that predicts player position N frames ahead from (x, y, vx, vy).
Input: normalized [x/W, y/H, vx/speed, vy/speed].
Output: normalized [pred_x/W, pred_y/H] after PREDICT_FRAMES.
"""
import numpy as np

# Match game
W = 900
H = 600
PLAYER_SPEED = 4
PREDICT_FRAMES = 14

INPUT_DIM = 4
HIDDEN = 32
OUTPUT_DIM = 2


def _wrap_norm(px: float, py: float) -> tuple:
    """Wrap to [0, 1] normalized."""
    while px < 0: px += 1
    while px > 1: px -= 1
    while py < 0: py += 1
    while py > 1: py -= 1
    return px, py


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PlayerPredictor(nn.Module):
        def __init__(self, input_dim: int = INPUT_DIM, hidden: int = HIDDEN, out_dim: int = OUTPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

else:
    PlayerPredictor = None  # type: ignore
