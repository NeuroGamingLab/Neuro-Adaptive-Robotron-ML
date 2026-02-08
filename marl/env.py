"""
Multi-agent RL environment for Robotron-style coordination.
Mirrors game/index.html physics: wrap, walls, player, grunts, bullets.
Used for centralized training; each agent gets local observation.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Match game constants
W = 900
H = 600
PLAYER_RADIUS = 12
PLAYER_SPEED = 4
GRUNT_RADIUS = 14
GRUNT_SPEED = 1.2
GRUNT_FIRE_COOLDOWN_MS = 1500
PLAYER_BULLET_SPEED = 14
ENEMY_BULLET_SPEED = 6
PREDICT_FRAMES = 14
SMOOTH_FACTOR = 0.88
OBS_SCALE = 500.0  # normalize positions by this
MAX_ALLIES = 10
N_WALL_RAYS = 4
WALL_RAY_LEN = 150


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _circle_rect_overlap(cx: float, cy: float, cr: float, r: Dict) -> bool:
    closest_x = np.clip(cx, r["x"], r["x"] + r["w"])
    closest_y = np.clip(cy, r["y"], r["y"] + r["h"])
    return _dist(cx, cy, closest_x, closest_y) < cr


def _wrap(x: float, y: float, radius: float) -> Tuple[float, float]:
    nx, ny = x, y
    if nx - radius > W:
        nx -= W + 2 * radius
    if nx + radius < 0:
        nx += W + 2 * radius
    if ny - radius > H:
        ny -= H + 2 * radius
    if ny + radius < 0:
        ny += H + 2 * radius
    return nx, ny


class RobotronMARLEnv:
    """
    Multi-agent environment. Each robot is an agent with shared policy.
    State: player, predicted player, walls, other robots (aggregated).
    Actions: 9 movement directions + fire (binary).
    Reward: team reward (damage player +, robot died -, small time penalty).
    """

    N_MOVE_ACTIONS = 9   # 8 dirs + stay
    N_FIRE_ACTIONS = 2   # no fire / fire
    OBS_DIM = 14         # see _get_obs

    def __init__(
        self,
        n_enemies: int = 8,
        n_walls: int = 8,
        max_steps: int = 2000,
        seed: Optional[int] = None,
    ):
        self.n_enemies = n_enemies
        self.n_walls = n_walls
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # Movement: 0=stay, 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE
        self._move_dirs = np.array([
            (0, 0), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)
        ], dtype=np.float32)
        self._move_dirs = self._move_dirs / (np.linalg.norm(self._move_dirs, axis=1, keepdims=True) + 1e-8)
        self._move_dirs[0] = 0
        self.reset()

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        self.step_count = 0
        self.player_health = 3
        self.player_lives = 3
        self.player_x = W / 2
        self.player_y = H / 2
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.avg_player_vx = 0.0
        self.avg_player_vy = 0.0
        self.pred_player_x = self.player_x
        self.pred_player_y = self.player_y
        self.walls = self._gen_walls()
        self.enemies = self._spawn_enemies()
        self.bullets: List[Dict] = []
        self.last_damage_to_player = 0
        self.last_robot_deaths = 0
        obs, infos = self._get_obs_all()
        return obs, infos

    def _gen_walls(self) -> List[Dict]:
        walls = []
        margin = 60
        for _ in range(self.n_walls):
            ww = 28 + int(self.rng.random() * 32)
            wh = 32 + int(self.rng.random() * 40)
            wx = int(self.rng.random() * (W - ww - 40)) + 20
            wy = int(self.rng.random() * (H - wh - 40)) + 20
            if _dist(wx + ww / 2, wy + wh / 2, W / 2, H / 2) < 80:
                continue
            walls.append({"x": wx, "y": wy, "w": ww, "h": wh})
        return walls

    def _spawn_enemies(self) -> List[Dict]:
        enemies = []
        margin = 60
        for i in range(self.n_enemies):
            side = self.rng.integers(0, 4)
            if side == 0:
                x = margin + self.rng.random() * (W - 2 * margin)
                y = -GRUNT_RADIUS - 5
            elif side == 1:
                x = W + GRUNT_RADIUS + 5
                y = margin + self.rng.random() * (H - 2 * margin)
            elif side == 2:
                x = margin + self.rng.random() * (W - 2 * margin)
                y = H + GRUNT_RADIUS + 5
            else:
                x = -GRUNT_RADIUS - 5
                y = margin + self.rng.random() * (H - 2 * margin)
            in_wall = any(
                _circle_rect_overlap(x, y, GRUNT_RADIUS + 4, w) for w in self.walls
            )
            if in_wall:
                x = np.clip(x, margin, W - margin)
                y = np.clip(y, margin, H - margin)
            enemies.append({
                "x": x, "y": y, "vx": 0.0, "vy": 0.0,
                "radius": GRUNT_RADIUS, "health": 1,
                "fire_cooldown_until": 0,
                "slot_index": i, "total_slots": self.n_enemies,
            })
        return enemies

    def _get_alpha(self) -> Optional[Dict]:
        alpha = None
        min_d = np.inf
        for e in self.enemies:
            d = _dist(e["x"], e["y"], self.player_x, self.player_y)
            if d < min_d:
                min_d = d
                alpha = e
        return alpha

    def _count_allies(self, e: Dict, exclude_self: bool = True) -> int:
        n = 0
        for other in self.enemies:
            if other is e:
                continue
            if _dist(e["x"], e["y"], other["x"], other["y"]) < 120:
                n += 1
        return n

    def _get_obs_one(self, e: Dict, alpha: Optional[Dict], now: float) -> np.ndarray:
        # Relative to player (normalized)
        dx_player = (self.player_x - e["x"]) / OBS_SCALE
        dy_player = (self.player_y - e["y"]) / OBS_SCALE
        dist_player = _dist(e["x"], e["y"], self.player_x, self.player_y) / OBS_SCALE
        # Relative to predicted player
        dx_pred = (self.pred_player_x - e["x"]) / OBS_SCALE
        dy_pred = (self.pred_player_y - e["y"]) / OBS_SCALE
        is_alpha = 1.0 if (alpha is not None and e is alpha) else 0.0
        allies = min(self._count_allies(e), MAX_ALLIES) / MAX_ALLIES
        fire_ready = 1.0 if now >= e["fire_cooldown_until"] else 0.0
        # Wall rays (4 cardinal directions)
        wall_dists = np.zeros(N_WALL_RAYS)
        for i in range(N_WALL_RAYS):
            ang = i * (2 * np.pi / N_WALL_RAYS)
            ex = e["x"] + np.cos(ang) * WALL_RAY_LEN
            ey = e["y"] + np.sin(ang) * WALL_RAY_LEN
            min_d = WALL_RAY_LEN
            for w in self.walls:
                # ray-rect (simplified: distance to rect center)
                cx = w["x"] + w["w"] / 2
                cy = w["y"] + w["h"] / 2
                d = _dist(e["x"], e["y"], cx, cy) - (w["w"] + w["h"]) / 4
                if d < min_d:
                    min_d = max(0, d)
            wall_dists[i] = min_d / WALL_RAY_LEN
        # Player velocity (normalized)
        pvx = self.player_vx / (PLAYER_SPEED + 1e-8)
        pvy = self.player_vy / (PLAYER_SPEED + 1e-8)
        obs = np.array([
            dx_player, dy_player, dist_player,
            dx_pred, dy_pred,
            is_alpha, allies, fire_ready,
            wall_dists[0], wall_dists[1], wall_dists[2], wall_dists[3],
            pvx, pvy,
        ], dtype=np.float32)
        return obs

    def _get_obs_all(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        alpha = self._get_alpha()
        now = self.step_count * 16.0  # ms
        obs_list = [
            self._get_obs_one(e, alpha, now) for e in self.enemies
        ]
        obs = {
            "obs": np.stack(obs_list, axis=0) if obs_list else np.zeros((0, self.OBS_DIM), dtype=np.float32),
            "n_agents": len(self.enemies),
        }
        infos = {"alpha": alpha}
        return obs, infos

    def _step_player(self, action: Optional[Tuple[int, int]] = None):
        # Simple scripted player: random walk or stay (for training)
        if action is None:
            if self.rng.random() < 0.3:
                dx = self.rng.choice([-1, 0, 1])
                dy = self.rng.choice([-1, 0, 1])
                if dx != 0 or dy != 0:
                    d = np.sqrt(dx * dx + dy * dy)
                    self.player_vx = dx / d * PLAYER_SPEED
                    self.player_vy = dy / d * PLAYER_SPEED
            else:
                self.player_vx *= 0.9
                self.player_vy *= 0.9
        else:
            dx, dy = action
            if dx != 0 or dy != 0:
                d = np.sqrt(dx * dx + dy * dy)
                self.player_vx = dx / d * PLAYER_SPEED
                self.player_vy = dy / d * PLAYER_SPEED
            else:
                self.player_vx *= 0.9
                self.player_vy *= 0.9
        prev_x, prev_y = self.player_x, self.player_y
        self.player_x += self.player_vx
        self.player_y += self.player_vy
        self.player_x, self.player_y = _wrap(self.player_x, self.player_y, PLAYER_RADIUS)
        for w in self.walls:
            if _circle_rect_overlap(self.player_x, self.player_y, PLAYER_RADIUS, w):
                self.player_x, self.player_y = prev_x, prev_y
                break
        self.avg_player_vx = SMOOTH_FACTOR * self.avg_player_vx + (1 - SMOOTH_FACTOR) * self.player_vx
        self.avg_player_vy = SMOOTH_FACTOR * self.avg_player_vy + (1 - SMOOTH_FACTOR) * self.player_vy
        self.pred_player_x = self.player_x + self.avg_player_vx * PREDICT_FRAMES
        self.pred_player_y = self.player_y + self.avg_player_vy * PREDICT_FRAMES

    def _step_enemies(self, move_actions: np.ndarray, fire_actions: np.ndarray, now: float):
        alpha = self._get_alpha()
        for i, e in enumerate(self.enemies):
            if i >= len(move_actions):
                continue
            move_idx = int(move_actions[i]) % self.N_MOVE_ACTIONS
            fire = fire_actions[i] > 0.5 if i < len(fire_actions) else False
            dx, dy = self._move_dirs[move_idx]
            e["vx"] = dx * GRUNT_SPEED
            e["vy"] = dy * GRUNT_SPEED
            old_x, old_y = e["x"], e["y"]
            e["x"] += e["vx"]
            e["y"] += e["vy"]
            e["x"], e["y"] = _wrap(e["x"], e["y"], e["radius"])
            for w in self.walls:
                if _circle_rect_overlap(e["x"], e["y"], e["radius"], w):
                    e["x"], e["y"] = old_x, old_y
                    break
            if fire and now >= e["fire_cooldown_until"]:
                bdx = self.pred_player_x - e["x"]
                bdy = self.pred_player_y - e["y"]
                bd = np.sqrt(bdx * bdx + bdy * bdy) + 1e-8
                self.bullets.append({
                    "x": e["x"] + (bdx / bd) * (e["radius"] + 4),
                    "y": e["y"] + (bdy / bd) * (e["radius"] + 4),
                    "vx": (bdx / bd) * ENEMY_BULLET_SPEED,
                    "vy": (bdy / bd) * ENEMY_BULLET_SPEED,
                    "radius": 4,
                    "is_player": False,
                })
                e["fire_cooldown_until"] = now + GRUNT_FIRE_COOLDOWN_MS

    def _step_bullets(self) -> Tuple[int, int]:
        damage_to_player = 0
        robots_killed = 0
        dt = 16.0
        to_remove = []
        for i, b in enumerate(self.bullets):
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            if b["x"] < -50 or b["x"] > W + 50 or b["y"] < -50 or b["y"] > H + 50:
                to_remove.append(i)
                continue
            for w in self.walls:
                if w["x"] <= b["x"] <= w["x"] + w["w"] and w["y"] <= b["y"] <= w["y"] + w["h"]:
                    to_remove.append(i)
                    break
            if b.get("is_player"):
                for j, e in enumerate(self.enemies):
                    if _dist(b["x"], b["y"], e["x"], e["y"]) < 4 + e["radius"]:
                        e["health"] -= 1
                        damage_to_player += 0
                        robots_killed += 1
                        to_remove.append(i)
                        break
            else:
                if _dist(b["x"], b["y"], self.player_x, self.player_y) < 4 + PLAYER_RADIUS:
                    damage_to_player += 1
                    self.player_health -= 1
                    to_remove.append(i)
        for i in reversed(to_remove):
            self.bullets.pop(i)
        return damage_to_player, robots_killed

    def _remove_dead_enemies(self) -> int:
        n = 0
        self.enemies = [e for e in self.enemies if e["health"] > 0]
        return n

    def step(
        self,
        move_actions: np.ndarray,
        fire_actions: np.ndarray,
        player_action: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, bool, Dict]:
        """
        move_actions: (n_agents,) int in [0, N_MOVE_ACTIONS)
        fire_actions: (n_agents,) float in [0,1] or binary
        Returns: obs, rewards, terminated, truncated, infos
        """
        now = self.step_count * 16.0
        self._step_player(player_action)
        self._step_enemies(move_actions, fire_actions, now)
        damage, robots_killed = self._step_bullets()
        self._remove_dead_enemies()
        # Check player-enemy collision
        for e in self.enemies:
            if _dist(self.player_x, self.player_y, e["x"], e["y"]) < PLAYER_RADIUS + e["radius"]:
                self.player_health -= 1
                damage += 1
                e["health"] -= 1
        self._remove_dead_enemies()
        # Team reward: damage to player is good for enemies, robot death bad
        r = 0.1 * damage - 0.5 * robots_killed - 0.001
        rewards = np.full(len(self.enemies), r, dtype=np.float32)
        self.step_count += 1
        terminated = self.player_health <= 0 or len(self.enemies) == 0
        truncated = self.step_count >= self.max_steps
        obs, infos = self._get_obs_all()
        infos["damage_to_player"] = damage
        infos["robots_killed"] = robots_killed
        return obs, rewards, terminated, truncated, infos
