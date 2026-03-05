"""
train_belief_net.py
───────────────────
Simulates GongZhu games, collects per-step belief snapshots, and trains the
GongZhuBeliefPredictor.

PSRO update
───────────
train() now accepts an optional `policy_pool` argument.  When supplied,
each game is played by 4 agents freshly sampled from the pool, so the
belief network learns to infer hidden cards from the play patterns of actual
trained policies rather than purely random noise.

When policy_pool is None, the original hardcoded agent list is used, keeping
the script fully functional as a standalone pre-training tool.

train() also accepts `resume_from` so the PSRO orchestrator can continue
training the same belief model across generations rather than starting from
scratch each time.

Agents
──────
Any callable of the form

    agent(env: GongZhuEnv) -> int

that returns a valid card index is compatible.  RandomAgent and EpsilonAgent
live in policy_pool.py; import them from there.

Loss
────
CrossEntropyLoss(ignore_index=-100) on raw safe_logits; already-played cards
(y_true=-100) are silently skipped.

LR schedule
───────────
Linear warm-up for warmup_fraction of gradient steps, then cosine decay to 0.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from base_env import GongZhuEnv
from belief_net import GongZhuBeliefPredictor
from policy_pool import RandomAgent, EpsilonAgent


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int = 100_000):
        self.buffer:   list     = []
        self.max_size: int      = max_size
        self._idx:     int      = 0

    def add(self, snapshot: Dict) -> None:
        if len(self.buffer) < self.max_size:
            self.buffer.append(snapshot)
        else:
            self.buffer[self._idx] = snapshot
            self._idx = (self._idx + 1) % self.max_size

    def sample(self, n: int) -> List[Dict]:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Batch collation
# ──────────────────────────────────────────────────────────────────────────────

def collate_batch(snapshots: List[Dict], device: torch.device) -> tuple:
    B     = len(snapshots)
    max_T = max(max(s["seq_len"] for s in snapshots), 1)

    played_cards_np   = np.zeros((B, max_T), dtype=np.int64)
    players_np        = np.zeros((B, max_T), dtype=np.int64)
    trick_nums_np     = np.zeros((B, max_T), dtype=np.int64)
    trick_pos_np      = np.zeros((B, max_T), dtype=np.int64)
    seq_lengths_np    = np.zeros(B,          dtype=np.int64)
    revealed_hands_np = np.zeros((B, 4, 52), dtype=np.float32)
    mask_np           = np.zeros((B, 52, 4), dtype=np.float32)
    y_true_np         = np.full((B, 52), -100, dtype=np.int64)

    for i, s in enumerate(snapshots):
        T = s["seq_len"]
        seq_lengths_np[i] = T
        if T > 0:
            played_cards_np[i, :T] = s["played_cards"]
            players_np[i,      :T] = s["players"]
            trick_nums_np[i,   :T] = s["trick_nums"]
            trick_pos_np[i,    :T] = s["trick_pos"]
        revealed_hands_np[i] = s["revealed_hands"]
        mask_np[i]           = s["mask"]
        y_true_np[i]         = s["y_true"]

    return (
        torch.from_numpy(played_cards_np).to(device),
        torch.from_numpy(players_np).to(device),
        torch.from_numpy(trick_nums_np).to(device),
        torch.from_numpy(trick_pos_np).to(device),
        torch.from_numpy(revealed_hands_np).to(device),
        torch.from_numpy(mask_np).to(device),
        torch.from_numpy(seq_lengths_np).to(device),
        torch.from_numpy(y_true_np).to(device),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Game simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_game(
    env:         GongZhuEnv,
    agents:      List[Callable],
    observers:   Optional[List[int]] = None,
    subset_prob: float = 1.0,
) -> List[Dict]:
    """
    Simulate one complete game and collect training snapshots.

    A snapshot is taken BEFORE each card is played (representing the task
    "given history so far, predict where each remaining card is").
    This yields up to 52 × len(observers) snapshots per call.

    Args:
        env:         A GongZhuEnv that has already been reset.
        agents:      List of 4 callables indexed by player.
        observers:   Player indices to collect from.  Defaults to all 4.
        subset_prob: Probability of retaining each individual snapshot.

    Returns:
        List of snapshot dicts.
    """
    if observers is None:
        observers = [0, 1, 2, 3]

    snapshots: List[Dict] = []

    while not env.done:
        if subset_prob >= 1.0 or random.random() < subset_prob:
            for obs in observers:
                snapshots.append(env.get_snapshot(obs))
        player = env.current_player
        action = agents[player](env)
        env.step(action)

    return snapshots


# ──────────────────────────────────────────────────────────────────────────────
# LR scheduler
# ──────────────────────────────────────────────────────────────────────────────

def get_warmup_cosine_schedule(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    # ── PSRO ──────────────────────────────────────────────────────────────────
    policy_pool       = None,     # PolicyPool | None
    resume_from:      Optional[str] = None,   # Path to checkpoint to resume from

    # ── Volume ────────────────────────────────────────────────────────────────
    num_games:        int   = 100_000,
    games_per_loop:   int   = 10,
    updates_per_loop: int   = 10,
    subset_prob:      float = 0.5,
    observers:        Optional[List[int]] = None,

    # ── Model ─────────────────────────────────────────────────────────────────
    d_model:  int = 128,
    n_heads:  int = 4,
    n_layers: int = 3,

    # ── Optimisation ──────────────────────────────────────────────────────────
    batch_size:      int   = 256,
    lr:              float = 2e-4,
    weight_decay:    float = 1e-4,
    warmup_fraction: float = 0.10,
    grad_clip:       float = 1.0,

    # ── Buffer ────────────────────────────────────────────────────────────────
    buffer_size:   int = 100_000,
    buffer_warmup: int = 10_000,

    # ── Logging / checkpoints ─────────────────────────────────────────────────
    log_every:  int = 500,
    save_every: int = 5_000,
    save_dir:   str = "ckpt",

    # ── Hardware ──────────────────────────────────────────────────────────────
    device_str: Optional[str] = None,
) -> GongZhuBeliefPredictor:
    """
    Train the belief network.  Returns the trained model.

    PSRO usage
    ──────────
    Pass `policy_pool` to make each simulated game use agents sampled from
    the current population.  Pass `resume_from` to continue training the
    same model across PSRO generations instead of re-initialising weights.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[belief] Training on {device}")

    os.makedirs(save_dir, exist_ok=True)
    if observers is None:
        observers = [0, 1, 2, 3]

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = GongZhuBeliefPredictor(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loops   = (num_games + games_per_loop - 1) // games_per_loop
    total_updates = total_loops * updates_per_loop
    warmup_steps  = max(1, int(warmup_fraction * total_updates))
    scheduler     = get_warmup_cosine_schedule(optimizer, warmup_steps, total_updates)

    updates_done = 0

    # ── Optional checkpoint resume ────────────────────────────────────────────
    if resume_from is not None and os.path.exists(resume_from):
        print(f"[belief] Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        # scheduler.load_state_dict(ckpt["sched_state"]) <- commented out for stability
        updates_done = ckpt.get("updates_done", 0)
        print(f"[belief] Resumed at update {updates_done}")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[belief] Fresh model. Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer = ReplayBuffer(max_size=buffer_size)

    # ── Default agents (used when no pool is provided) ────────────────────────
    _base           = RandomAgent()
    _default_agents: List[Callable] = [
        RandomAgent(),                           # Player 0 — pure random
        EpsilonAgent(_base, epsilon=0.15),       # Player 1 — 15 % random
        EpsilonAgent(_base, epsilon=0.05),       # Player 2 —  5 % random
        RandomAgent(),                           # Player 3 — pure random
    ]

    env = GongZhuEnv()

    # ── Buffer warmup ─────────────────────────────────────────────────────────
    print(f"[belief] Warming up buffer to {buffer_warmup} snapshots...")
    while len(buffer) < buffer_warmup:
        agents = (
            policy_pool.sample_opponents(n=4)
            if policy_pool is not None
            else _default_agents
        )
        env.reset()
        for snap in simulate_game(env, agents, observers=observers,
                                   subset_prob=subset_prob):
            buffer.add(snap)
    print("[belief] Warmup complete. Starting training.")

    # ── Tracking ──────────────────────────────────────────────────────────────
    running_loss  = 0.0
    running_steps = 0
    t0            = time.time()

    run_hid_corr   = 0; run_hid_tot   = 0
    run_early_corr = 0; run_early_tot = 0
    run_late_corr  = 0; run_late_tot  = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for game_idx in range(num_games):

        # 1. Sample agents for this game
        #    PSRO: draw from the pool so the belief net learns real play patterns.
        #    Standalone: use the default random/epsilon agents.
        agents = (
            policy_pool.sample_opponents(n=4)
            if policy_pool is not None
            else _default_agents
        )

        # 2. Simulate and store
        env.reset()
        for snap in simulate_game(env, agents, observers=observers,
                                   subset_prob=subset_prob):
            buffer.add(snap)

        # 3. Gradient updates every games_per_loop games
        if (game_idx + 1) % games_per_loop == 0:
            model.train()

            for _ in range(updates_per_loop):
                batch = buffer.sample(batch_size)
                (
                    played_cards, players, trick_nums, trick_pos,
                    revealed_hands, mask, seq_lengths, y_true,
                ) = collate_batch(batch, device)

                _, safe_logits = model(
                    played_cards, players, trick_nums, trick_pos,
                    revealed_hands, mask, seq_lengths=seq_lengths,
                )
                loss = criterion(
                    safe_logits.view(-1, 4),
                    y_true.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                running_loss  += loss.item()
                running_steps += 1
                updates_done  += 1

                # Evaluation metrics
                with torch.no_grad():
                    preds      = safe_logits.argmax(dim=-1)
                    valid_mask = (y_true != -100)
                    is_hidden  = (mask > -1e8).sum(dim=-1) > 1
                    eval_mask  = valid_mask & is_hidden

                    correct_guesses = (preds == y_true) & eval_mask
                    run_hid_corr += correct_guesses.sum().item()
                    run_hid_tot  += eval_mask.sum().item()

                    seq_len_exp = seq_lengths.unsqueeze(1).expand_as(eval_mask)
                    early_mask  = eval_mask & (seq_len_exp >= 8)  & (seq_len_exp < 24)
                    late_mask   = eval_mask & (seq_len_exp >= 28) & (seq_len_exp < 44)

                    run_early_corr += (correct_guesses & early_mask).sum().item()
                    run_early_tot  += early_mask.sum().item()
                    run_late_corr  += (correct_guesses & late_mask).sum().item()
                    run_late_tot   += late_mask.sum().item()

                if updates_done % log_every == 0:
                    elapsed   = time.time() - t0
                    avg_loss  = running_loss / running_steps
                    cur_lr    = scheduler.get_last_lr()[0]
                    hid_acc   = (run_hid_corr   / run_hid_tot)   * 100 if run_hid_tot   > 0 else 0.0
                    early_acc = (run_early_corr / run_early_tot) * 100 if run_early_tot > 0 else 0.0
                    late_acc  = (run_late_corr  / run_late_tot)  * 100 if run_late_tot  > 0 else 0.0

                    pool_info = (
                        f"  Pool={len(policy_pool)}" if policy_pool is not None else ""
                    )
                    print(
                        f"[{updates_done:6d}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {cur_lr:.2e} | "
                        f"Acc(Hid): {hid_acc:.1f}% | "
                        f"(Early: {early_acc:.1f}% - Late: {late_acc:.1f}%) | "
                        f"Elapsed: {elapsed:.0f}s"
                        f"{pool_info}"
                    )

                    running_loss = 0.0;  running_steps = 0
                    run_hid_corr = 0;    run_hid_tot   = 0
                    run_early_corr = 0;  run_early_tot  = 0
                    run_late_corr  = 0;  run_late_tot   = 0

        # 4. Checkpoint
        if (game_idx + 1) % save_every == 0:
            path = os.path.join(save_dir, f"belief_game{game_idx + 1}.pt")
            torch.save(
                {
                    "game":         game_idx + 1,
                    "updates_done": updates_done,
                    "model_state":  model.state_dict(),
                    "optim_state":  optimizer.state_dict(),
                    "sched_state":  scheduler.state_dict(),
                    "config": dict(
                        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        lr=lr, batch_size=batch_size,
                    ),
                },
                path,
            )
            print(f"  ↳ checkpoint saved to {path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "belief_final.pt")
    torch.save(
        {
            "game":         num_games,
            "updates_done": updates_done,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "sched_state":  scheduler.state_dict(),
            "config": dict(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                lr=lr, batch_size=batch_size,
            ),
        },
        final_path,
    )
    print(f"\n[belief] Training complete. Final model saved to {final_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GongZhuBeliefPredictor")

    p.add_argument("--num-games",        type=int,   default=100_000)
    p.add_argument("--games-per-loop",   type=int,   default=10)
    p.add_argument("--updates-per-loop", type=int,   default=10)
    p.add_argument("--subset-prob",      type=float, default=0.5)

    p.add_argument("--d-model",  type=int, default=128)
    p.add_argument("--n-heads",  type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)

    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--grad-clip",       type=float, default=1.0)

    p.add_argument("--buffer-size",   type=int, default=100_000)
    p.add_argument("--buffer-warmup", type=int, default=10_000)

    p.add_argument("--log-every",  type=int, default=500)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--save-dir",   type=str, default="ckpt")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to belief checkpoint to resume from.")

    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        policy_pool      = None,
        resume_from      = args.resume_from,
        num_games        = args.num_games,
        games_per_loop   = args.games_per_loop,
        updates_per_loop = args.updates_per_loop,
        subset_prob      = args.subset_prob,
        d_model          = args.d_model,
        n_heads          = args.n_heads,
        n_layers         = args.n_layers,
        batch_size       = args.batch_size,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        warmup_fraction  = args.warmup_fraction,
        grad_clip        = args.grad_clip,
        buffer_size      = args.buffer_size,
        buffer_warmup    = args.buffer_warmup,
        log_every        = args.log_every,
        save_every       = args.save_every,
        save_dir         = args.save_dir,
        device_str       = args.device,
    )