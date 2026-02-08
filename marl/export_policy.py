"""
Export trained MARL policy to JSON for in-browser inference.
Format: list of layers with W (weights) and b (bias) as nested arrays.
"""
import json
import os

from .policy import MARLPolicy, OBS_DIM, N_MOVE, N_FIRE

try:
    import torch
except ImportError:
    torch = None


def export_to_json(state_dict_path: str, out_path: str) -> None:
    if torch is None:
        raise RuntimeError("PyTorch required for export. pip install torch")
    policy = MARLPolicy()
    policy.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    policy.eval()

    def tensor_to_list(t):
        return t.detach().cpu().numpy().tolist()

    # Order: net.0 (linear), net.2 (linear), move_head, fire_head
    data = {
        "obs_dim": OBS_DIM,
        "n_move": N_MOVE,
        "n_fire": N_FIRE,
        "hidden": 64,
        "hidden2": 64,
        "layers": [
            {
                "W": tensor_to_list(policy.net[0].weight),
                "b": tensor_to_list(policy.net[0].bias),
            },
            {
                "W": tensor_to_list(policy.net[2].weight),
                "b": tensor_to_list(policy.net[2].bias),
            },
        ],
        "move_head": {
            "W": tensor_to_list(policy.move_head.weight),
            "b": tensor_to_list(policy.move_head.bias),
        },
        "fire_head": {
            "W": tensor_to_list(policy.fire_head.weight),
            "b": tensor_to_list(policy.fire_head.bias),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Exported policy to {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="marl_checkpoints/policy.pt")
    p.add_argument("--out", type=str, default="game/marl_policy.json")
    args = p.parse_args()
    export_to_json(args.ckpt, args.out)


if __name__ == "__main__":
    main()
