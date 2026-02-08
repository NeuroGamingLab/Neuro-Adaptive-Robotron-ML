"""
Simple DNN: player position predictor.
Predicts future player position (e.g. 14 frames ahead) from current position and velocity.
"""
from .predictor import PlayerPredictor, INPUT_DIM, PREDICT_FRAMES, W, H, PLAYER_SPEED

__all__ = ["PlayerPredictor", "INPUT_DIM", "PREDICT_FRAMES", "W", "H", "PLAYER_SPEED"]
