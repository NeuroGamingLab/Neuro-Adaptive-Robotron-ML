"""
Export trained DNN predictor to JSON for in-browser use.
"""
import json
import os

from .predictor import PlayerPredictor, INPUT_DIM, HIDDEN, OUTPUT_DIM

try:
    import torch
except ImportError:
    torch = None


def export_to_json(state_dict_path: str, out_path: str) -> None:
    if torch is None:
        raise RuntimeError("PyTorch required for export. pip install torch")
    model = PlayerPredictor()
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.eval()

    def to_list(t):
        return t.detach().cpu().numpy().tolist()

    # Two linear layers: net.0 (input->hidden), net.2 (hidden->out)
    data = {
        "input_dim": INPUT_DIM,
        "hidden": HIDDEN,
        "output_dim": OUTPUT_DIM,
        "layers": [
            {"W": to_list(model.net[0].weight), "b": to_list(model.net[0].bias)},
            {"W": to_list(model.net[2].weight), "b": to_list(model.net[2].bias)},
        ],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Exported DNN predictor to {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="dnn_checkpoints/predictor.pt")
    p.add_argument("--out", type=str, default="game/dnn_predictor.json")
    args = p.parse_args()
    export_to_json(args.ckpt, args.out)


if __name__ == "__main__":
    main()
