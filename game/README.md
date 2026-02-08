# Robotron with ML

Multidirectional shooter: move with arrow keys, shoot with Space (aim = arrow keys held when firing).

## How to play

- **Arrow keys** — Move (8-direction with diagonals)
- **Space** — Fire in the direction of the arrow key(s) you're holding. Hold one direction to move, tap another + Space to shoot that way.
- Clear each wave of robots; they move toward you and shoot back. Avoid bullets and body contact.

## Run locally

**Option 1 — Open directly:** Double-click `index.html` or open it from your browser (File → Open). The game uses inlined CSS and JS so it loads even when opened via `file://`.

**Option 2 — Streamlit (from project root):**
```bash
cd /path/to/robotron-ml
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501 — click the game area so arrow keys and Space work
```

**Option 3 — Local HTTP server (optional):**
```bash
cd game
python3 -m http.server 8080
# Open http://localhost:8080
```

## Structure

- `index.html` — Canvas and UI
- `css/style.css` — Retro green-on-black styling
- `js/game.js` — Game loop, player, enemies, bullets, waves, collision
- **`ENEMY-PATTERNS.md`** — How enemy robots move and attack (flanking, alpha, gap-fill, predictive aim, staggered fire, MARL)
