"""
Train the simple DNN player predictor on synthetic random-walk data.
Target: predict position 14 frames ahead with constant-velocity assumption + wrap.
"""
import argparse
import os

import numpy as np

from .predictor import (
    W,
    H,
    PLAYER_SPEED,
    PREDICT_FRAMES,
    INPUT_DIM,
    OUTPUT_DIM,
    _wrap_norm,
    PlayerPredictor,
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_data(n_samples: int, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, INPUT_DIM), dtype=np.float32)
    Y = np.zeros((n_samples, OUTPUT_DIM), dtype=np.float32)
    for i in range(n_samples):
        x = rng.uniform(0, 1)
        y = rng.uniform(0, 1)
        vx = rng.uniform(-1, 1)
        vy = rng.uniform(-1, 1)
        X[i] = [x, y, vx, vy]
        # Target: position after PREDICT_FRAMES steps at constant velocity (normalized)
        # In world: px = x*W + PREDICT_FRAMES * vx*PLAYER_SPEED, then wrap
        px = (x * W + PREDICT_FRAMES * vx * PLAYER_SPEED) / W
        py = (y * H + PREDICT_FRAMES * vy * PLAYER_SPEED) / H
        px, py = _wrap_norm(px, py)
        Y[i] = [px, py]
    return X, Y


def train(
    n_samples: int = 50000,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-2,
    save_dir: str = "dnn_checkpoints",
    seed: int = 42,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required. pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    X, Y = generate_data(n_samples, seed=seed)
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlayerPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={total_loss/n_batches:.6f}")

    path = os.path.join(save_dir, "predictor.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")
    return model, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--save-dir", type=str, default="dnn_checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
