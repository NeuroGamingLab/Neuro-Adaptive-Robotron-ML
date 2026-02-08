/**
 * Robotron with ML — Multidirectional shooter
 * Arrow keys: move. Space: fire in direction of arrow keys held.
 */

(function () {
  'use strict';

  const canvas = document.getElementById('game-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  const UI = {
    scoreEl: document.getElementById('score'),
    livesEl: document.getElementById('lives'),
    waveEl: document.getElementById('wave'),
    startScreen: document.getElementById('start-screen'),
    gameOverScreen: document.getElementById('game-over-screen'),
    waveClearScreen: document.getElementById('wave-clear-screen'),
    finalScoreEl: document.getElementById('final-score'),
    startBtn: document.getElementById('start-btn'),
    restartBtn: document.getElementById('restart-btn'),
    nextWaveBtn: document.getElementById('next-wave-btn'),
  };

  // ——— Constants ———
  const PLAYER = {
    radius: 12,
    speed: 4,
    health: 3,
    lives: 3,
    invincibleDuration: 1500,
    fireCooldownMs: 150,
  };
  const PLAYER_BULLET = { radius: 3, speed: 14 };
  const GRUNT = {
    radius: 14,
    speed: 1.2,
    health: 1,
    fireCooldownMs: 1500,
    bulletSpeed: 6,
    score: 100,
  };
  const ENEMY_BULLET = { radius: 4, speed: 6 };
  const WAVE_COUNTS = [3, 5, 8, 12, 15];
  const WAVE_CLEAR_PAUSE_MS = 1500;

  // ——— State ———
  let state = 'start';
  let score = 0;
  let wave = 1;
  let waveClearTimer = 0;

  let player = {
    x: W / 2,
    y: H / 2,
    vx: 0,
    vy: 0,
    health: PLAYER.health,
    lives: PLAYER.lives,
    invincibleUntil: 0,
    lastAimDx: 0,
    lastAimDy: -1,
    fireCooldownUntil: 0,
  };

  let bullets = [];
  let enemies = [];
  let keys = { up: false, down: false, left: false, right: false, space: false };

  // ——— Input ———
  function getMoveDirs() {
    let dx = 0, dy = 0;
    if (keys.up) dy -= 1;
    if (keys.down) dy += 1;
    if (keys.left) dx -= 1;
    if (keys.right) dx += 1;
    if (dx !== 0 || dy !== 0) {
      const len = Math.sqrt(dx * dx + dy * dy);
      dx /= len;
      dy /= len;
    }
    return { dx, dy };
  }

  function getAimDirs() {
    let dx = 0, dy = 0;
    if (keys.up) dy -= 1;
    if (keys.down) dy += 1;
    if (keys.left) dx -= 1;
    if (keys.right) dx += 1;
    if (dx !== 0 || dy !== 0) {
      const len = Math.sqrt(dx * dx + dy * dy);
      dx /= len;
      dy /= len;
      return { dx, dy };
    }
    return null;
  }

  document.addEventListener('keydown', function (e) {
    if (e.code === 'ArrowUp') { keys.up = true; e.preventDefault(); }
    if (e.code === 'ArrowDown') { keys.down = true; e.preventDefault(); }
    if (e.code === 'ArrowLeft') { keys.left = true; e.preventDefault(); }
    if (e.code === 'ArrowRight') { keys.right = true; e.preventDefault(); }
    if (e.code === 'Space') { keys.space = true; e.preventDefault(); }
  });
  document.addEventListener('keyup', function (e) {
    if (e.code === 'ArrowUp') keys.up = false;
    if (e.code === 'ArrowDown') keys.down = false;
    if (e.code === 'ArrowLeft') keys.left = false;
    if (e.code === 'ArrowRight') keys.right = false;
    if (e.code === 'Space') keys.space = false;
  });

  // ——— Spawn ———
  function spawnPlayer() {
    player.x = W / 2;
    player.y = H / 2;
    player.vx = 0;
    player.vy = 0;
    player.health = PLAYER.health;
    player.invincibleUntil = Date.now() + PLAYER.invincibleDuration;
    player.lastAimDx = 0;
    player.lastAimDy = -1;
    player.fireCooldownUntil = 0;
  }

  function spawnWave() {
    const count = WAVE_COUNTS[Math.min(wave - 1, WAVE_COUNTS.length - 1)];
    for (let i = 0; i < count; i++) {
      const side = Math.floor(Math.random() * 4);
      let x, y;
      const margin = 60;
      if (side === 0) { x = margin + Math.random() * (W - 2 * margin); y = -GRUNT.radius - 5; }
      else if (side === 1) { x = W + GRUNT.radius + 5; y = margin + Math.random() * (H - 2 * margin); }
      else if (side === 2) { x = margin + Math.random() * (W - 2 * margin); y = H + GRUNT.radius + 5; }
      else { x = -GRUNT.radius - 5; y = margin + Math.random() * (H - 2 * margin); }
      enemies.push({
        x, y,
        vx: 0, vy: 0,
        radius: GRUNT.radius,
        health: GRUNT.health,
        type: 'grunt',
        fireCooldownUntil: Date.now() + Math.random() * 800,
      });
    }
  }

  // ——— Collision ———
  function dist(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  }

  function wrap(x, y, r) {
    let nx = x, ny = y;
    if (nx - r > W) nx -= W + 2 * r;
    if (nx + r < 0) nx += W + 2 * r;
    if (ny - r > H) ny -= H + 2 * r;
    if (ny + r < 0) ny += H + 2 * r;
    return { x: nx, y: ny };
  }

  // ——— Update ———
  function updatePlayer(now) {
    const { dx, dy } = getMoveDirs();
    player.vx = dx * PLAYER.speed;
    player.vy = dy * PLAYER.speed;
    player.x += player.vx;
    player.y += player.vy;
    const w = wrap(player.x, player.y, PLAYER.radius);
    player.x = w.x;
    player.y = w.y;

    if (keys.space && now >= player.fireCooldownUntil) {
      const aim = getAimDirs();
      let adx = player.lastAimDx, ady = player.lastAimDy;
      if (aim) {
        adx = aim.dx;
        ady = aim.dy;
        player.lastAimDx = adx;
        player.lastAimDy = ady;
      }
      if (adx !== 0 || ady !== 0) {
        const spd = PLAYER_BULLET.speed;
        bullets.push({
          x: player.x + adx * (PLAYER.radius + PLAYER_BULLET.radius),
          y: player.y + ady * (PLAYER.radius + PLAYER_BULLET.radius),
          vx: adx * spd,
          vy: ady * spd,
          radius: PLAYER_BULLET.radius,
          isPlayer: true,
        });
        player.fireCooldownUntil = now + PLAYER.fireCooldownMs;
      }
    }
  }

  function updateBullets(dt) {
    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i];
      b.x += b.vx;
      b.y += b.vy;
      if (b.x < -50 || b.x > W + 50 || b.y < -50 || b.y > H + 50) {
        bullets.splice(i, 1);
      }
    }
  }

  function updateEnemies(now) {
    for (const e of enemies) {
      const dx = player.x - e.x;
      const dy = player.y - e.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      e.vx = (dx / d) * GRUNT.speed;
      e.vy = (dy / d) * GRUNT.speed;
      e.x += e.vx;
      e.y += e.vy;
      const w = wrap(e.x, e.y, e.radius);
      e.x = w.x;
      e.y = w.y;

      if (e.type === 'grunt' && now >= e.fireCooldownUntil) {
        const bdx = player.x - e.x;
        const bdy = player.y - e.y;
        const bd = Math.sqrt(bdx * bdx + bdy * bdy) || 1;
        const vx = (bdx / bd) * GRUNT.bulletSpeed;
        const vy = (bdy / bd) * GRUNT.bulletSpeed;
        bullets.push({
          x: e.x + (bdx / bd) * (e.radius + ENEMY_BULLET.radius),
          y: e.y + (bdy / bd) * (e.radius + ENEMY_BULLET.radius),
          vx, vy,
          radius: ENEMY_BULLET.radius,
          isPlayer: false,
        });
        e.fireCooldownUntil = now + GRUNT.fireCooldownMs;
      }
    }
  }

  function checkCollisions(now) {
    const invincible = now < player.invincibleUntil;

    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i];
      if (b.isPlayer) {
        for (let j = enemies.length - 1; j >= 0; j--) {
          const e = enemies[j];
          if (dist(b.x, b.y, e.x, e.y) < b.radius + e.radius) {
            e.health--;
            bullets.splice(i, 1);
            if (e.health <= 0) {
              score += GRUNT.score;
              enemies.splice(j, 1);
            }
            break;
          }
        }
      } else {
        if (!invincible && dist(b.x, b.y, player.x, player.y) < b.radius + PLAYER.radius) {
          player.health--;
          bullets.splice(i, 1);
        }
      }
    }

    if (!invincible) {
      for (let j = enemies.length - 1; j >= 0; j--) {
        const e = enemies[j];
        if (dist(player.x, player.y, e.x, e.y) < PLAYER.radius + e.radius) {
          player.health--;
          e.health--;
          if (e.health <= 0) {
            score += GRUNT.score;
            enemies.splice(j, 1);
          }
        }
      }
    }
  }

  function update(now) {
    if (state === 'wave_clear') {
      waveClearTimer -= 16;
      if (waveClearTimer <= 0) {
        state = 'playing';
        waveClearTimer = 0;
        UI.waveClearScreen.classList.add('hidden');
        spawnWave();
      }
      return;
    }
    if (state !== 'playing') return;

    updatePlayer(now);
    updateBullets(16);
    updateEnemies(now);
    checkCollisions(now);

    if (player.health <= 0) {
      player.lives--;
      bullets = [];
      if (player.lives <= 0) {
        state = 'game_over';
        UI.gameOverScreen.classList.remove('hidden');
        UI.finalScoreEl.textContent = 'Score: ' + score;
      } else {
        spawnPlayer();
      }
    }

    if (enemies.length === 0 && state === 'playing') {
      state = 'wave_clear';
      waveClearTimer = WAVE_CLEAR_PAUSE_MS;
      UI.waveClearScreen.classList.remove('hidden');
      wave++;
    }
  }

  // ——— Draw ———
  function drawPlayer(now) {
    ctx.save();
    if (now < player.invincibleUntil) {
      ctx.globalAlpha = 0.5 + 0.5 * Math.sin(now / 80);
    }
    ctx.fillStyle = '#0f0';
    ctx.beginPath();
    ctx.arc(player.x, player.y, PLAYER.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0a0';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  function drawBullets() {
    for (const b of bullets) {
      if (b.isPlayer) {
        const len = 24;
        let dx = b.vx, dy = b.vy;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        dx /= d; dy /= d;
        const x0 = b.x - dx * len, y0 = b.y - dy * len;
        const x1 = b.x + dx * len, y1 = b.y + dy * len;
        ctx.save();
        ctx.strokeStyle = '#0ff';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#0ff';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.fillStyle = '#f80';
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawEnemies() {
    for (const e of enemies) {
      ctx.fillStyle = '#c00';
    ctx.beginPath();
      ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#800';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function draw() {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    if (state === 'playing' || state === 'wave_clear') {
      drawBullets();
      drawEnemies();
      drawPlayer(Date.now());
    }

    if (state === 'wave_clear' && waveClearTimer > 0) {
      UI.waveEl.textContent = 'Wave ' + (wave - 1) + ' clear! Next: ' + wave;
    }
  }

  function tick() {
    update(Date.now());
    draw();
    if (state === 'playing' || state === 'wave_clear') {
      UI.scoreEl.textContent = 'Score: ' + score;
      UI.livesEl.textContent = 'Lives: ' + player.lives;
      UI.waveEl.textContent = 'Wave: ' + wave;
    }
    requestAnimationFrame(tick);
  }

  // ——— Start / Restart ———
  function startGame() {
    state = 'playing';
    score = 0;
    wave = 1;
    bullets = [];
    enemies = [];
    player.health = PLAYER.health;
    player.lives = PLAYER.lives;
    player.invincibleUntil = 0;
    UI.startScreen.classList.add('hidden');
    UI.gameOverScreen.classList.add('hidden');
    UI.waveClearScreen.classList.add('hidden');
    spawnPlayer();
    spawnWave();
  }

  function nextWave() {
    state = 'playing';
    waveClearTimer = 0;
    UI.waveClearScreen.classList.add('hidden');
    spawnWave();
  }

  UI.startBtn.addEventListener('click', startGame);
  UI.restartBtn.addEventListener('click', startGame);
  UI.nextWaveBtn.addEventListener('click', nextWave);

  requestAnimationFrame(tick);
})();
