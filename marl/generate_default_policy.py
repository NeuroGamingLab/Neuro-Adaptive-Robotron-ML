"""
Generate a default (small random) MARL policy JSON for in-game use.
Use this to test MARL mode before training. After training, use export_policy to overwrite.
"""
import json
import os
import sys

from .policy import OBS_DIM, N_MOVE, N_FIRE

HIDDEN = 64
HIDDEN2 = 64


def generate(out_path: str, seed: int = 42) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
        def rand(): return float(np.random.randn() * 0.1)
    except ImportError:
        import random
        rng = random.Random(seed)
        def rand(): return (rng.random() - 0.5) * 0.2

    def mat(in_dim, out_dim):
        return [[rand() for _ in range(in_dim)] for _ in range(out_dim)]

    def vec(dim):
        return [rand() for _ in range(dim)]

    data = {
        "obs_dim": OBS_DIM,
        "n_move": N_MOVE,
        "n_fire": N_FIRE,
        "hidden": HIDDEN,
        "hidden2": HIDDEN2,
        "layers": [
            {"W": mat(OBS_DIM, HIDDEN), "b": vec(HIDDEN)},
            {"W": mat(HIDDEN, HIDDEN2), "b": vec(HIDDEN2)},
        ],
        "move_head": {"W": mat(HIDDEN2, N_MOVE), "b": vec(N_MOVE)},
        "fire_head": {"W": mat(HIDDEN2, N_FIRE), "b": vec(N_FIRE)},
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote default policy to {out_path}")


def main():
    out = "game/marl_policy.json"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    generate(out)


if __name__ == "__main__":
    main()
