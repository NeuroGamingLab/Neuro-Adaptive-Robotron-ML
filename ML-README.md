# ML Features — Robotron with ML

This document describes all machine-learning and ML-inspired features used by the robots in the game.

---

## Overview

The game uses a mix of **predictive** (smoothed velocity, lead target), **coordination** (formation, alpha, staggered fire), and **unsupervised** (angular clustering, gap-filling) logic so robots find and trap the player and coordinate attacks. Rule-based and unsupervised parts run in-browser each frame. **Optional:** a **Simple DNN** can predict player position (replacing EMA), and a **MARL** policy can drive enemy movement and fire when trained and loaded as JSON; both run in-browser from exported weights.

---

## 1. Player movement prediction

**Purpose:** Anticipate where the player will be so robots can move and aim ahead.

- **Smoothed velocity (exponential moving average)**  
  Each frame:
  - `avgPlayerVx = smoothFactor * avgPlayerVx + (1 - smoothFactor) * player.vx`
  - Same for `avgPlayerVy`.  
  Reduces jitter from sudden direction changes.

- **Predicted position**  
  - `predictedPlayerX = player.x + avgPlayerVx * predictFrames`
  - `predictedPlayerY = player.y + avgPlayerVy * predictFrames`  
  Robots use this point for formation center and aim.

**Parameters (in `ML`):**

| Parameter       | Default | Role                                      |
|----------------|---------|-------------------------------------------|
| `smoothFactor` | 0.88    | How much to smooth velocity (higher = smoother) |
| `predictFrames`| 14      | How many frames ahead to predict          |

**Reset:** Smoothed velocity and predicted position are reset when the game starts (e.g. on “Start Game”).

---

## 2. Formation & flanking

**Purpose:** Robots surround the player (or alpha) instead of all moving to the same point.

- **Formation center:** Either the **alpha robot’s position** or, if none, the **predicted player position**.
- **Slot angles:** Each robot has a slot index; target angle = `formationPhase + (slotIndex / totalSlots) * 2π`.
- **Formation phase:** A slow global rotation: `(now * formationPhaseSpeed) % 2π`, so the formation orbits over time.
- **Dynamic radius:** Formation radius depends on ally count (see §5). Target position = center + (cos(angle), sin(angle)) × radius.

**Parameters:**

| Parameter            | Default | Role                          |
|---------------------|---------|-------------------------------|
| `flankRadius`       | 140     | Base formation radius (px)    |
| `flankRadiusMin`    | 100     | Minimum radius               |
| `flankRadiusMax`    | 180     | Maximum radius               |
| `formationPhaseSpeed` | 0.00015 | Rotation speed of formation |

---

## 3. Alpha robot (leader)

**Purpose:** One robot “leads” the pack and closes in; others coordinate around it.

- **Definition:** The robot **closest to the player** (by distance) is the alpha.
- **Visual:** Alpha is drawn **yellow**; others **red**.
- **Behavior:**  
  - Alpha moves straight toward the **predicted player position**.  
  - Non-alpha robots use the **alpha’s current position** as the formation center and move to flank positions around it (or use the unsupervised pipeline target angle; see §6).

Recomputed every frame, so the alpha can change as distances change (e.g. when the previous alpha is destroyed).

---

## 4. Predictive aim (lead target)

**Purpose:** Shoot where the player will be, not where they are.

- When a robot fires, it aims at **predicted player position** `(predictedPlayerX, predictedPlayerY)`.
- Bullet direction = from robot to that point, normalized.
- Shots land better when the player keeps moving in a steady direction.

---

## 5. Staggered fire

**Purpose:** Spread shots over time so the player faces a stream of fire from different directions instead of one big volley.

- Each robot gets a **phase offset** at spawn: `(slotIndex % 8) * 220` ms plus a random offset.
- First shot is delayed by that phase; subsequent shots use the normal fire cooldown.
- Result: fire is distributed in time and space.

---

## 6. Tactical radius (ally-based)

**Purpose:** Adjust formation radius from local density (more allies → hold back, fewer → close in).

- For each robot, **allies** = number of other robots within `allyRadius` (120 px).
- **Dynamic radius:**  
  `radius = flankRadius + (allies - 2) * 18`, clamped to `[flankRadiusMin, flankRadiusMax]`.
- So: many allies → larger radius (spread out); few allies → smaller radius (close in).

**Parameter:**

| Parameter    | Default | Role                    |
|-------------|---------|-------------------------|
| `allyRadius`| 120     | Range to count allies   |

---

## 7. Unsupervised ML pipeline (find & trap)

**Purpose:** Discover “gaps” (escape routes) around the player and assign robots to fill them so they trap the player. No labels; structure is inferred from robot positions only.

### 7.1 Angular clustering

- **Center:** Predicted player position `(predictedPlayerX, predictedPlayerY)`.
- **Feature per robot:** Angle from center to robot: `atan2(robot.y - cy, robot.x - cx)` in `[0, 2π)`.
- **Bins:** `nAngularBins` (12) equal angular bins. Each robot counted in one bin by its angle.
- **Unsupervised:** Only positions are used; no labels.

### 7.2 Gap detection

- **Gap bin:** Any bin with count ≤ `gapThreshold` (default 1).
- **Interpretation:** Sectors with few or no robots = escape routes.
- **Gap centers:** Mid-angle of each gap bin: `(binIndex + 0.5) * (2π / nAngularBins)`.

### 7.3 Assignment (fill gaps)

- Robots are **sorted by current angle** (order around the player).
- Gap centers are **sorted by angle**.
- **Round-robin:** Robot at sorted index `i` is assigned target angle = `gapCenters[i % numGaps]`.
- Non-alpha robots move toward that **target angle** at the formation radius (around alpha or predicted player).
- **Effect:** Robots spread into underpopulated sectors and close escape routes.

### 7.4 Fallback

- If **no gaps** (all bins above threshold), target angles fall back to the slot-based formation (see §2).

**Parameters:**

| Parameter       | Default | Role                                  |
|----------------|---------|---------------------------------------|
| `nAngularBins` | 12      | Number of angular sectors             |
| `gapThreshold` | 1       | Max count in a bin to count as gap    |
| `crowdThreshold` | 2    | Reserved (e.g. for “crowded” logic)  |

---

## 8. Visual: lines to player

**Purpose:** Show each robot’s intended “move toward the player” direction.

- A **line** is drawn from each robot’s center to the player’s center.
- Alpha: yellow line; others: light red.  
  Purely visual; no effect on ML logic.

---

## 9. Where it lives in code (rule-based / unsupervised)

- **Game logic:** `game/index.html` (inlined script).
- **ML constants:** Object `ML` and globals `avgPlayerVx`, `avgPlayerVy`, `predictedPlayerX`, `predictedPlayerY`.
- **Pipeline:** `runUnsupervisedPipeline()` (angular clustering, gap detection, assignment); called each frame from `updateEnemies()`.
- **Design doc:** `DESIGN.md` §8 and §8.1.

---

## 10. Multi-agent RL (MARL) for robot coordination

**Purpose:** Train robot policies with MARL so they coordinate (flanking, covering, etc.) instead of relying only on hand-coded formation logic. When a trained policy is loaded, it replaces the rule-based formation/unsupervised pipeline for enemy movement and fire.

### 10.1 Design

- **Setup:** Shared policy (one network, same weights for all robots). Centralized training with team reward; decentralized execution (each robot gets its own observation and chooses an action).
- **State (per-robot observation, 14 dims):** Relative position to player and to predicted player (normalized); distance to player; is-alpha (0/1); ally count (normalized); fire cooldown ready; 4 wall-ray distances; player velocity. All normalized so the policy generalizes.
- **Actions:** **Move** — 9 discrete (stay, 8 directions); **Fire** — binary (no / yes). Movement is applied as velocity; fire uses the same lead-target aim (predicted player position) as the rule-based version when the policy chooses to fire.
- **Reward (training):** Team reward: damage to player (+), robot deaths (−), small time penalty. All agents get the same reward each step to encourage coordination.

### 10.2 Training

- **Environment:** Python env in `marl/env.py` mirrors game physics (wrap, walls, player, grunts, bullets). Player is scripted (e.g. random walk) during training.
- **Algorithm:** PPO (shared policy) in `marl/train_ppo.py`. Each step, each robot gets its observation and samples move + fire from the policy; rewards are shared.
- **Commands:**
  - Train: `python -m marl.train_ppo [--n-enemies 8] [--steps 300000] [--save-dir marl_checkpoints]`
  - Export: `python -m marl.export_policy --ckpt marl_checkpoints/policy.pt --out game/marl_policy.json`
  - Default policy (random, for testing MARL mode): `python -m marl.generate_default_policy` → writes `game/marl_policy.json`

### 10.3 Game integration

- **Policy file:** `game/marl_policy.json` (weights only: two hidden layers + move head + fire head). If this file exists and the game is run via the Streamlit app, it is injected as `window.MARL_POLICY`.
- **In-game:** In `updateEnemies()`, if `window.MARL_POLICY` is set, each robot gets `getMARLObsOne()`, then `MARLForward()` returns move and fire; velocity and firing are applied. Otherwise the existing formation + unsupervised pipeline is used.
- **No extra runtime:** Inference is a small MLP in plain JS (no TensorFlow.js). One forward pass per robot per frame.

### 10.4 Where it lives

- **Training:** `marl/env.py`, `marl/policy.py`, `marl/train_ppo.py`, `marl/export_policy.py`, `marl/generate_default_policy.py`
- **Game:** `game/index.html` — `MARL_CONST`, `getMARLObsOne()`, `MARLForward()`, and the MARL branch in `updateEnemies()`
- **App:** `app.py` injects `game/marl_policy.json` as `window.MARL_POLICY` when the file exists

---

## 11. Simple DNN (player position predictor)

**Purpose:** A small feedforward network that predicts future player position (e.g. 14 frames ahead) from current position and velocity. When loaded, it replaces the EMA-based extrapolation for `predictedPlayerX` / `predictedPlayerY`, which all enemy logic (formation center, aim, MARL obs) uses. This is a minimal “first DNN” you can add before or alongside MARL.

### 11.1 Design

- **Input (4 dims):** Normalized `[x/W, y/H, vx/PLAYER_SPEED, vy/PLAYER_SPEED]`.
- **Output (2 dims):** Normalized predicted position `[pred_x/W, pred_y/H]` after 14 frames (sigmoid so in [0, 1]).
- **Architecture:** Two layers: linear → tanh (hidden 32) → linear → sigmoid (2). No recurrence; one forward pass per frame.

### 11.2 Training and export

- **Data:** Synthetic: random (x, y, vx, vy) in normalized space; target = position after 14 steps at constant velocity, wrapped to [0, 1].
- **Train:** `python -m dnn.train [--samples 50000] [--epochs 50] [--save-dir dnn_checkpoints]`
- **Export:** `python -m dnn.export_policy --ckpt dnn_checkpoints/predictor.pt --out game/dnn_predictor.json`
- **Default (no training):** `python -m dnn.generate_default` → writes `game/dnn_predictor.json` so DNN prediction can be used immediately (behavior similar to extrapolation once trained).

### 11.3 Game integration

- **When `window.DNN_PREDICT` is set:** In the game loop, `predictedPlayerX` / `predictedPlayerY` are set from `DNNPredict(xNorm, yNorm, vxNorm, vyNorm)` (then scaled by W, H). Otherwise the existing EMA + linear extrapolation is used.
- **Where:** `game/index.html` — `DNNPredict()`, and the branch in `update()` that sets predicted position. `app.py` injects `game/dnn_predictor.json` as `window.DNN_PREDICT` when the file exists.

---

## 12. Summary table

| Feature                 | Type / idea           | Role                                      |
|-------------------------|-----------------------|-------------------------------------------|
| Smoothed velocity       | Exponential moving avg| Stable prediction of player movement      |
| Predicted position      | Extrapolation         | Where to form up and aim                  |
| **Simple DNN**          | Feedforward NN        | Learned player position predictor (replaces EMA when loaded) |
| Formation & flanking     | Coordination          | Surround player/alpha                     |
| Alpha robot             | Leader selection      | One robot leads, others form around it   |
| Predictive aim          | Lead target           | Shoot at predicted position               |
| Staggered fire          | Phase offset          | Spread shots in time                      |
| Tactical radius         | Ally-based rule       | Adjust formation radius from density      |
| Unsupervised pipeline   | Clustering + assignment| Find gaps, fill them to trap player      |
| **MARL**                | Multi-agent RL (PPO)  | Learned coordination; replaces rules when policy loaded |

---

*ML features for Robotron with ML — NeuroGamingLab · Designer & architect: Tuệ Hoàng, AI/ML Engineer*
*Multi-LLM–assisted development.*
