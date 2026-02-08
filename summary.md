# Robotron with ML — Summary

## What it is

**Robotron with ML** is a browser-based multidirectional shooter inspired by Robotron. You move in one direction and shoot in another while fighting waves of robots that move, form up, and shoot back. Enemy behavior combines rule-based logic with optional ML (player prediction and learned coordination).

---

## Core gameplay

- **Player:** One character on a top-down arena. Move with arrow keys; aim with the arrow key(s) held when you press Fire. Move one way, shoot another.
- **Enemies:** Robots (e.g. Grunts) advance toward you, form around a “leader” (alpha), and fire at your predicted position. They use formations, staggered fire, and gap-filling to trap you.
- **Waves:** Clear all robots to finish a wave; next wave spawns. Game over when lives (or health) are gone.
- **Extras:** Smart missile (homes on nearest robot), temporary shield, and destructible walls add variety.

---

## Controls

| Action        | Input                          |
|--------------|---------------------------------|
| Move         | **↑↓←→** Arrow keys             |
| Aim          | Arrow key(s) held when firing   |
| Fire         | **Space**                       |
| Smart missile| **S**                           |
| Shield (hold)| **F**                           |
| New walls    | **G**                           |
| Start        | **Enter** / Click               |
| Pause        | **P** / **Escape**              |

*Click the game area first so keys are captured.*

---

## Tech stack

- **Frontend:** HTML5 + JavaScript; game runs from a single `game/index.html` (no heavy ML runtime). Optional DNN and MARL run as small MLPs in JS from exported JSON weights.
- **Backend / launcher:** Python **Streamlit** (`app.py`) embeds the game and injects optional MARL/DNN policy files when present.
- **ML (Python):** **PyTorch** for training. **DNN** predicts player position; **MARL** (PPO) trains a shared policy for robot move/fire. Both export to JSON for in-browser inference.

---

## ML at a glance

- **Prediction:** Smoothed player velocity + optional DNN to predict where you’ll be; robots aim and form up around that point.
- **Coordination:** Formation/flanking, alpha (closest robot) leading, staggered fire, ally-based radius. Optional **unsupervised-style** gap-filling so robots close escape routes.
- **Optional overrides:**  
  - `game/dnn_predictor.json` — use a trained DNN instead of EMA for player position.  
  - `game/marl_policy.json` — use a trained MARL policy for robot move/fire instead of pure rules.

See **ML-README.md** for parameters and training commands.

---

## How to run

1. **Streamlit (recommended):**  
   `source .venv/bin/activate` → `pip install -r requirements.txt` → `streamlit run app.py` → open **http://localhost:8501**
2. **Script:**  
   `./start.sh` (from project root, with executable venv + Streamlit).
3. **Docker:**  
   `docker build -t robotron-ml .` and `docker run -p 8501:8501 robotron-ml`
4. **Game only:**  
   Open `game/index.html` in a browser (no MARL/DNN injection).

---

## Docs

- **README.md** — Setup, run options, project layout.
- **DESIGN.md** — Full game design (player, enemies, waves, controls).
- **ML-README.md** — All ML features, training, and export.

**NeuroGamingLab** · Design & architecture: Tuệ Hoàng, AI/ML Engineer.  
*Multi-LLM–assisted development.*

