"""
Generate a default (approx linear) DNN predictor JSON so the game can use DNN prediction before training.
Weights are set so output ≈ input position + constant * velocity (rough extrapolation).
"""
import json
import os

from .predictor import INPUT_DIM, HIDDEN, OUTPUT_DIM, PREDICT_FRAMES, W, H, PLAYER_SPEED

# Approx: pred_x ≈ x + k*vx, pred_y ≈ y + k*vy with k = PREDICT_FRAMES*PLAYER_SPEED/W for x and /H for y
# In normalized: pred_x_norm = x + (PREDICT_FRAMES*PLAYER_SPEED/W)*vx = x + 14*4/900*vx ≈ x + 0.062*vx
# So we want layer weights to approximate identity for x,y and small scale for vx,vy. Sigmoid output [0,1].
# Simple: first layer random small, second layer: out[0] gets mostly x + 0.06*vx, out[1] gets mostly y + 0.06*vy.
# For a minimal default we just put small random weights so it doesn't explode; game falls back to EMA if bad.
def generate(out_path: str, seed: int = 42) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
        def rand(): return float(np.random.randn() * 0.1)
    except ImportError:
        import random
        rng = random.Random(seed)
        def rand(): return (rng.random() - 0.5) * 0.2

    def mat(rows, cols):
        return [[rand() for _ in range(cols)] for _ in range(rows)]
    def vec(n):
        return [rand() for _ in range(n)]

    data = {
        "input_dim": INPUT_DIM,
        "hidden": HIDDEN,
        "output_dim": OUTPUT_DIM,
        "layers": [
            {"W": mat(HIDDEN, INPUT_DIM), "b": vec(HIDDEN)},
            {"W": mat(OUTPUT_DIM, HIDDEN), "b": vec(OUTPUT_DIM)},
        ],
    }
    # Bias second layer so sigmoid(0)≈0.5: set b to small values so output is near 0.5 without input
    for i in range(OUTPUT_DIM):
        data["layers"][1]["b"][i] = 0.0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote default DNN predictor to {out_path}")


def main():
    import sys
    out = "game/dnn_predictor.json"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    generate(out)


if __name__ == "__main__":
    main()
