"""
Train MARL policy with PPO (shared policy, team reward).
Saves checkpoint and exports to JSON for game.
"""
import argparse
import json
import os
import numpy as np

from .env import RobotronMARLEnv
from .policy import MARLPolicy, OBS_DIM, N_MOVE, N_FIRE

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train(
    n_enemies: int = 8,
    n_walls: int = 8,
    total_timesteps: int = 300_000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    save_dir: str = "marl_checkpoints",
    seed: int = 42,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training. Install with: pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    env = RobotronMARLEnv(n_enemies=n_enemies, n_walls=n_walls, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MARLPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    obs, _ = env.reset()
    n_agents = obs["n_agents"]
    obs_buf = np.zeros((n_steps, n_enemies, OBS_DIM), dtype=np.float32)
    move_act_buf = np.zeros((n_steps, n_enemies), dtype=np.int64)
    fire_act_buf = np.zeros((n_steps, n_enemies), dtype=np.int64)
    reward_buf = np.zeros((n_steps, n_enemies), dtype=np.float32)
    done_buf = np.zeros((n_steps,), dtype=np.float32)
    value_buf = np.zeros((n_steps, n_enemies), dtype=np.float32)
    log_prob_buf = np.zeros((n_steps, n_enemies), dtype=np.float32)

    global_step = 0
    ep_returns = []
    ep_len = 0
    ep_ret = 0.0

    while global_step < total_timesteps:
        for step in range(n_steps):
            if obs["n_agents"] == 0:
                obs, _ = env.reset()
            o = obs["obs"]
            n_agents = o.shape[0]
            t_obs = torch.from_numpy(o).to(device)
            with torch.no_grad():
                move_act, fire_act, log_prob, _, value = policy.get_action_and_value(t_obs)
            move_np = move_act.cpu().numpy()
            fire_np = fire_act.cpu().numpy()
            # Pad to n_enemies for storage
            obs_buf[step, :n_agents] = o
            obs_buf[step, n_agents:] = 0
            move_act_buf[step, :n_agents] = move_np
            move_act_buf[step, n_agents:] = 0
            fire_act_buf[step, :n_agents] = fire_np
            fire_act_buf[step, n_agents:] = 0
            value_buf[step, :n_agents] = value.cpu().numpy()
            value_buf[step, n_agents:] = 0
            log_prob_buf[step, :n_agents] = log_prob.cpu().numpy()
            log_prob_buf[step, n_agents:] = 0

            obs, rewards, term, trunc, _ = env.step(move_np, fire_np)
            reward_buf[step, :n_agents] = rewards
            reward_buf[step, n_agents:] = 0
            done_buf[step] = 1.0 if (term or trunc) else 0.0
            ep_ret += rewards.sum() / max(n_agents, 1)
            ep_len += 1
            global_step += n_agents

            if term or trunc:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                ep_len = 0
                obs, _ = env.reset()

        # GAE
        next_obs = obs["obs"]
        next_n = next_obs.shape[0]
        with torch.no_grad():
            next_value = policy.get_value(torch.from_numpy(next_obs).to(device)).cpu().numpy()
        next_value_pad = np.zeros(n_enemies, dtype=np.float32)
        next_value_pad[:next_n] = next_value
        advantages = np.zeros((n_steps, n_enemies), dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - done_buf[t]
                nextvalues = next_value_pad
            else:
                nextnonterminal = 1.0
                nextvalues = value_buf[t + 1]
            delta = reward_buf[t] + gamma * nextvalues * nextnonterminal - value_buf[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + value_buf

        # Flatten for minibatch (only use valid agents)
        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_move = move_act_buf.reshape(-1)
        b_fire = fire_act_buf.reshape(-1)
        b_log_prob = log_prob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        # Mask: only update where we had real agents (obs was not all zeros)
        valid = np.abs(b_obs).sum(axis=1) > 1e-6
        idx = np.where(valid)[0]
        if len(idx) == 0:
            obs, _ = env.reset()
            continue
        n_batches = (len(idx) + batch_size - 1) // batch_size

        for _ in range(n_epochs):
            perm = np.random.permutation(len(idx))
            for start in range(0, len(idx), batch_size):
                end = start + batch_size
                mb = idx[perm[start:end]]
                mb_obs = torch.from_numpy(b_obs[mb]).to(device)
                mb_move = torch.from_numpy(b_move[mb]).long().to(device)
                mb_fire = torch.from_numpy(b_fire[mb]).long().to(device)
                mb_adv = torch.from_numpy(b_adv[mb]).to(device)
                mb_ret = torch.from_numpy(b_ret[mb]).to(device)
                mb_old_log_prob = torch.from_numpy(b_log_prob[mb]).to(device)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                _, _, new_log_prob, entropy, new_value = policy.get_action_and_value(
                    mb_obs, move_action=mb_move, fire_action=mb_fire
                )
                ratio = (new_log_prob - mb_old_log_prob).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (new_value - mb_ret).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + ent_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if len(ep_returns) > 0 and global_step % 10000 < n_steps * n_enemies:
            print(f"step {global_step} return {ep_returns[-1]:.2f}")

    # Save
    ckpt_path = os.path.join(save_dir, "policy.pt")
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return policy, ckpt_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-enemies", type=int, default=8)
    p.add_argument("--n-walls", type=int, default=8)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="marl_checkpoints")
    args = p.parse_args()
    train(
        n_enemies=args.n_enemies,
        n_walls=args.n_walls,
        total_timesteps=args.steps,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
