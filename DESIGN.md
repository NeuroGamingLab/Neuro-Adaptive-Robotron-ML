# Robotron with ML — Game Design Document

A multidirectional shooter where the player moves in one direction and shoots in another, fighting hordes of robot enemies that shoot back.

---

## 1. Core Pillar

- **Dual-stick feel (keyboard/mouse or keys):** Movement and aim are independent. Player runs one way, fires another.
- **Horde combat:** Many enemies at once. Tension comes from positioning and prioritization, not single duels.
- **Enemies shoot back:** Every robot is a threat. Dodging and cover matter; standing still is death.

---

## 2. Player

### 2.1 Movement
- **Arrow keys** for movement (4-direction or 8-direction with diagonals).
- **Constant move speed** (no acceleration). Responsive, arcade feel.
- **Screen wrap** or **solid borders** (design choice: wrap is more classic Robotron).

### 2.2 Aiming & Shooting
- **Fire:** **Space bar** shoots in the current aim direction.
- **Aim:** Same **arrow keys** as movement. Aim = direction of arrow key(s) **held at the moment you press Space**. So you can move one way and shoot another:
  - Hold **↑** to move up; tap **←** + **Space** to shoot left (then release ←); you keep moving up.
  - Hold **↑** + **→** to move up-right; press **Space** (with no other change) to shoot up-right.
- If no arrow key is held when Space is pressed, shoot in **last aim direction** (or last move direction).
- **Weapon:** Single gun that fires in the current aim direction.
  - **Fire rate:** Tuned so it feels powerful but not screen-clearing (e.g. 5–10 rounds/sec cap).
  - **Projectiles:** Instant or very fast; clear hitbox (e.g. small circle or line).

### 2.3 Health & Lives
- **Health:** e.g. 3–5 hits (or a small health bar).
- **Lives:** Optional. Classic: 3 lives, respawn in place or center; when 0 = game over.
- **Invulnerability:** Short invincibility after spawn/death (e.g. 1.5–2 s) so player can reorient.

### 2.4 Hitbox
- Small (e.g. circle or rectangle). Player should feel like a “dot” that can slip between shots.

---

## 3. Enemies (Robots)

### 3.1 Core Behavior
- **Move:** Toward player (or predictive) with simple, readable patterns.
- **Shoot:** At the player. Not every frame—cooldown so the screen isn’t a solid wall of bullets.
- **Collision:** Contact with player = damage to player (and optionally to enemy).

### 3.2 Enemy Types (suggested minimum)

| Type        | Speed   | Health | Fire rate      | Behavior summary        |
|------------|---------|--------|----------------|-------------------------|
| **Grunt**  | Slow    | 1      | Low            | Walks toward player, shoots occasionally. |
| **Runner** | Fast    | 1      | None or very low | Closes in, melee/collision threat. |
| **Turret** | None/very slow | 1–2 | High           | Shoots often, little movement. |
| **Heavy**  | Slow    | 3      | Medium         | Tank; soaks damage, steady fire. |

All types **shoot back** except possibly Runner (if you want a “melee only” variant). Tune so that:
- Bullets are visible and dodgeable.
- Screen doesn’t become unfair; spawn count and fire rates matter more than bullet speed.

### 3.3 Spawning
- **Wave-based:** Each wave has a set (or random within set) of enemy types and counts.
- Spawn at **screen edges** or from **spawn points**; avoid spawning on top of player.
- **Horde feel:** Wave size grows over time (e.g. 5 → 10 → 20+ robots).

---

## 4. Weapons & Combat

### 4.1 Player Bullets
- Clear sprite/line; small hitbox.
- One-shot **Grunt / Runner / Turret**; 2–3 hits for **Heavy** (or use health values).
- No friendly fire (no need for allies in v1).

### 4.2 Enemy Bullets
- Visually distinct from player bullets (e.g. color: red/orange).
- Damage: 1 player health per hit (or configurable).
- Easy to read so player can dodge; speed and rate tuned for fairness.

### 4.3 Collision
- **Player vs enemy bullet** → player hit, bullet removed.
- **Player bullet vs enemy** → enemy hit, bullet removed.
- **Player vs enemy body** → player damage (and optionally enemy damage or death).

---

## 5. Levels / Waves / Progression

- **Waves:** Wave 1, 2, 3… with increasing count and/or harder types.
- **Wave clear:** All enemies destroyed → short break → next wave.
- **End condition:**  
  - **Game over:** Player lives = 0 (or health = 0 if no lives).  
  - **Win:** Optional “final wave” or endless with high-score focus.

### 5.1 Scoring (optional but recommended)
- Points per kill (e.g. Grunt 100, Turret 200, Heavy 500).
- Bonus for wave clear, combo, or no damage (stretch).
- **High score** persisted (localStorage or backend later).

---

## 6. Controls (Locked)

| Action   | Input                  |
|----------|------------------------|
| Move     | **Arrow keys** (↑↓←→)  |
| Aim      | **Arrow keys** (direction held when firing) |
| Fire     | **Space bar**          |
| Start    | Enter / Click          |
| Pause    | P / Escape             |

**Move one way, shoot another:** Hold arrows to move; when you press Space, you shoot in the direction of the arrow key(s) held at that moment. Release an aim key after firing to keep moving without changing aim for the next shot.

---

## 7. Visual & Audio Style

- **Visual:** Top-down or slight angle. Clear silhouettes: player = one color, enemies = another, bullets distinct. Retro (pixel/vector) or simple shapes both work.
- **Audio:**  
  - Player shoot, enemy shoot, player hit, enemy death, wave start/end.  
  - Optional: background track or ambient.

---

## 8. Technical Approach

- **Platform:** Web (HTML5 + Canvas or WebGL). Fits template’s deployment (browser + optional Streamlit wrapper).
- **No modification** of `DONOT-MODIFY-template/game-template-only/`; build game in project root or a dedicated folder (e.g. `game/` or `robotron/`).
- **Deployment:** Reuse pattern from template: Terraform + EC2; serve static game or run a small server (e.g. Streamlit) that embeds or links to the game.
- **Optional ML:**  
  - Enemy AI: movement/aim tuned by simple ML or behavior params.  
  - Difficulty scaling: ML or rules that adapt spawn rate / bullet speed based on player performance.

### 8.1 Unsupervised ML pipeline (find & trap)

- **Angular clustering:** Each frame, robots are binned by angle around the predicted player position (12 bins). No labels; structure is discovered from the distribution of robot positions.
- **Gap detection:** Bins with count ≤ threshold are “gaps” (escape routes). Low-density directions around the player are identified without supervision.
- **Assignment:** Robots (sorted by current angle) are assigned to gap centers in round-robin so they spread to fill gaps. Each non-alpha robot gets a target angle; movement is toward that angle at formation radius.
- **Effect:** Robots self-organize to close escape routes and trap the player. Alpha still closes in; others fill gaps from the unsupervised assignment. Fallback to slot-based formation when no gaps exist.

### 8.2 Simple DNN (player position predictor)

- **Optional override:** When `game/dnn_predictor.json` is loaded, the game uses a small feedforward DNN to compute predicted player position (14 frames ahead) from current position and velocity, instead of the default EMA + linear extrapolation. All enemy logic (formation, aim, MARL) uses this predicted position.
- **Training:** Synthetic data (constant-velocity trajectories); export to JSON; in-browser inference. See `ML-README.md` §11.

### 8.3 Multi-agent RL (MARL) for robot coordination

- **Optional override:** When a trained MARL policy is loaded (`game/marl_policy.json`), enemy movement and fire are driven by the learned policy instead of the rule-based formation and unsupervised pipeline.
- **Training:** Python MARL env mirrors game physics; PPO with shared policy; team reward (damage player, survive). Policy is exported to JSON; the game runs a small MLP in JS for inference. See `ML-README.md` §10.

---

## 9. MVP Scope (Phase 1)

1. **Player:** 8-dir move, 1 weapon, aim (mouse or keys), health + lives.
2. **One enemy type:** Grunt—moves toward player, shoots back.
3. **Waves:** 3–5 waves, spawn at edges, wave clear = next wave.
4. **Collision:** Player/enemy bullets and body; damage and death.
5. **Game over / restart.**  
6. **Basic UI:** Score, lives/health, wave number, start and game over screens.

**Phase 2:** More enemy types (Runner, Turret, Heavy), more waves, sound, polish.  
**Phase 3:** High score, optional ML, deployment via Terraform.

---

## 10. File Structure (Suggested)

```
robotron-ml/
├── instruction.txt
├── DESIGN.md                 ← this file
├── DONOT-MODIFY-template/    ← do not modify
├── game/                     ← game implementation
│   ├── index.html
│   ├── css/
│   ├── js/
│   │   ├── main.js
│   │   ├── player.js
│   │   ├── enemies.js
│   │   ├── bullets.js
│   │   ├── waves.js
│   │   └── collision.js
│   └── assets/
│       ├── sprites/
│       └── audio/
├── terraform/                ← copy/adapt from template for deploy (optional)
└── README.md
```

---

## 11. Design Decisions to Lock In

**Decided:**
- **Controls:** Arrow keys = move; Space bar = fire; aim = arrow key(s) held when firing (same keys, independent direction).
- Document control scheme in README and in-game.

**Still to decide:**
1. **Screen wrap vs solid walls** for level bounds.
2. **Lives vs single health bar** (or both: health bar + lives).
3. **Enemy set for MVP:** Start with Grunt-only or Grunt + Runner.

---

*Design v1 — Robotron with ML*
