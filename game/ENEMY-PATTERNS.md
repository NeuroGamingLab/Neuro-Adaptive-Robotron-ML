# Enemy Robot Attack & Movement Patterns

This document describes how the enemy robots move and attack in **Robotron with ML**. They use formation, flanking, gap-filling, predictive aim, and staggered fire so the player is surrounded and under sustained fire from multiple directions.

---

## Overview

- **One enemy type in-game:** Grunt (slow, 1 HP, shoots at the player).
- **Two control modes:** **Rule-based** (formation + gap-fill + alpha) or **MARL** (learned policy when `marl_policy.json` is loaded). Aim is always predictive (lead target).
- **Visual states:** Alpha (yellow + ring + “ALPHA” label), ready-to-fire (brighter outline), on-cooldown (dimmer), filling-gap (cyan dashed ring), damaged (health bar + hit flash when multi-HP).

---

## 1. Formation & Flanking

**Goal:** Robots **surround** the player instead of all moving to the same spot.

- **Formation center:** Either the **alpha robot’s position** or, if there is no alpha, the **predicted player position**.
- **Target position (non-alpha):** A point on a **circle** around that center:
  - **Angle** = formation phase + slot angle, or the **gap-fill angle** when gaps exist (see §4).
  - **Radius** = formation radius (see §5). So: `target = center + (cos(angle), sin(angle)) × radius`.
- **Formation phase:** A slow global rotation so the formation **orbits** over time: `(time × formationPhaseSpeed) % 2π`.

**Tunable parameters (in `ML` in `index.html`):**

| Parameter             | Default | Role                          |
|-----------------------|---------|-------------------------------|
| `flankRadius`         | 140     | Base formation radius (px)    |
| `flankRadiusMin`      | 100     | Minimum radius                |
| `flankRadiusMax`      | 180     | Maximum radius                |
| `formationPhaseSpeed` | 0.00015 | Rotation speed of the ring    |

---

## 2. Alpha Robot (Leader)

**Goal:** One robot **leads** and closes in; the rest coordinate around it.

- **Who is alpha:** The robot **closest to the player** (by distance). Recomputed every frame (alpha can change when distances change or when the previous alpha is destroyed).
- **Alpha behavior:**
  - Moves **straight toward** the **predicted player position** (not the formation ring).
- **Non-alpha behavior:**
  - Use the **alpha’s current position** as the formation center and move to flank positions (or gap-fill angles) around it.

**Visual:** Alpha is drawn **yellow** with a ring and an **“ALPHA”** label above it.

---

## 3. Gap-Filling (Find & Trap)

**Goal:** Detect “gaps” (escape routes) around the player and **assign robots to fill them** so they tighten the ring and trap the player.

- **Angular bins:** 12 equal sectors around the **predicted player position**. Each robot is counted in one bin by its angle from that center.
- **Gap:** Any bin with count ≤ `gapThreshold` (default 1) is treated as a gap (low coverage = escape route).
- **Gap centers:** Mid-angle of each gap bin.
- **Assignment:** Non-alpha robots are **sorted by angle** around the player; gap centers are sorted by angle; robots are assigned to gap centers in **round-robin**. Each assigned robot gets a **target angle** and moves toward that angle at the formation radius.
- **Fallback:** If there are **no gaps** (all bins above threshold), target angles fall back to **slot-based formation** (even spread by slot index + formation phase).

**Visual:** Robots currently assigned to fill a gap show a **cyan dashed ring** (`isFillingGap`).

**Tunable parameters:**

| Parameter       | Default | Role                          |
|-----------------|---------|-------------------------------|
| `nAngularBins`  | 12      | Number of angular sectors     |
| `gapThreshold`  | 1       | Max count in a bin to be a gap|

---

## 4. Tactical Radius (Ally-Based)

**Goal:** Adjust formation radius from **local density**: more allies nearby → hold back (larger radius); fewer → close in (smaller radius).

- **Allies:** For each robot, count other robots within `allyRadius` (120 px).
- **Dynamic radius:** `radius = flankRadius + (allies - 2) × 18`, clamped to `[flankRadiusMin, flankRadiusMax]`.

**Tunable parameter:**

| Parameter    | Default | Role                  |
|-------------|---------|------------------------|
| `allyRadius`| 120     | Range to count allies  |

---

## 5. Predictive Aim (Lead Target)

**Goal:** Shoot **where the player will be**, not where they are.

- **Predicted position:** Either from **EMA + linear extrapolation** (`predictedPlayerX`, `predictedPlayerY`) or from the **Simple DNN** when `dnn_predictor.json` is loaded.
- **When a robot fires:** Bullet direction = from robot to **predicted player position**, normalized. Shots work better when the player keeps a steady direction.

---

## 6. Staggered Fire

**Goal:** Spread shots **over time** so the player faces a **stream of fire** from different directions instead of one big volley.

- **Per-robot phase:** At spawn, each robot gets a **fire phase offset**: `(slotIndex % 8) × 220` ms plus a random offset. The first shot is delayed by that phase; later shots use the normal fire cooldown.
- **Effect:** Fire is distributed in time and in space (because robots are in different positions).

**Tunable:** Grunt `fireCooldownMs` (e.g. 1500) in `GRUNT` in `index.html`.

---

## 7. MARL Mode (Optional)

When **`marl_policy.json`** is loaded (e.g. via the Streamlit app), **movement and fire decision** are driven by a **learned policy** instead of the rule-based formation and gap-fill logic.

- **Same:** Predictive aim (predicted player position) and game physics (speed, cooldowns, wrap, walls).
- **Different:** Each robot’s **move direction** and **whether to fire** come from the policy’s forward pass. Flanking-like behavior can emerge from training but is not hand-coded in MARL mode.
- **Visual:** Same states (alpha, ready-to-fire, on-cooldown, filling-gap, damaged) where applicable; `isFillingGap` is still set from the unsupervised pipeline so the “filling gap” ring can show even when movement is MARL-controlled.

---

## 8. Summary Table

| Pattern            | Role                                      |
|--------------------|-------------------------------------------|
| **Formation**      | Surround player/alpha at a radius         |
| **Flanking**       | Move to ring positions, not one point     |
| **Alpha**          | One leader closes in; others form around it|
| **Gap-filling**    | Fill angular gaps to trap player          |
| **Tactical radius**| Radius adapts to ally count               |
| **Predictive aim** | Shoot at predicted position               |
| **Staggered fire** | Phase-offset cooldowns for stream of fire |
| **MARL (optional)**| Learned move + fire instead of rules       |

---

## 9. Where It Lives in Code

- **Game logic:** `game/index.html` (inlined script).
- **Parameters:** `ML` and `GRUNT` objects; `predictedPlayerX`, `predictedPlayerY`.
- **Pipeline:** `runUnsupervisedPipeline()` (gap detection, assignment, `isFillingGap`); `updateEnemies()` (formation, alpha, movement, firing); MARL branch uses `getMARLObsOne()` and `MARLForward()` when `window.MARL_POLICY` is set.
- **Drawing:** `drawEnemies()` (alpha ring/label, ready-to-fire outline, on-cooldown dim, filling-gap ring, damaged bar/flash).

For full ML design (DNN predictor, MARL training, export), see the project’s **`ML-README.md`**.
