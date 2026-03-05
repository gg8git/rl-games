"""
run_psro.py  —  PSRO orchestrator for GongZhu (宫主)
═══════════════════════════════════════════════════════════════════════════════
Supports both a fully automated mode and a manual step-by-step mode so you
can evaluate checkpoints between generations and pick the best one before
committing it to the pool.

Subcommands
───────────
  init            Create a fresh pool state seeded with one RandomAgent.
                  Run this once before starting a new PSRO experiment.

  train-belief    Train (or resume) the belief network using all agents
                  currently in the pool.

  train-ppo       Train a PPO best response against the current pool.
                  All periodic checkpoints are saved so you can compare them
                  before deciding which one enters the pool.

  eval-meta-game  Simulate all unique 4-agent matchups, build the payoff
                  tensor, run AlphaRank, and write the meta-Nash distribution
                  back to pool_state.json.  Run after add-agent and before the
                  next train-belief / train-ppo so opponents are sampled
                  according to the meta-Nash rather than uniformly.
                  Requires OpenSpiel (pip install open_spiel); falls back to
                  uniform if unavailable.

  add-agent       Register a chosen PPO checkpoint into the pool.
                  Automatically clears the stale AlphaRank distribution so
                  the next training round uses uniform sampling until
                  eval-meta-game recomputes the weights.

  status          Print the current pool contents, AlphaRank weights, and
                  checkpoint locations.

  run-auto        Fully automated loop (belief → PPO → eval → add) for N
                  generations.  Pass --skip-eval to omit AlphaRank evaluation.

Pool state
──────────
All commands read/write  <save-dir>/pool_state.json, which stores the agent
descriptors and the AlphaRank distribution needed to reconstruct the
PolicyPool across Python sessions.  The belief checkpoint always lives at
<save-dir>/belief/belief_final.pt and is updated in-place by train-belief.

Typical manual workflow (with AlphaRank)
────────────────────────────────────────
  # One-time setup
  python run_psro.py init --save-dir psro_runs

  # Repeat for each generation:
  python run_psro.py train-belief    --save-dir psro_runs
  python run_psro.py train-ppo       --save-dir psro_runs

  # <-- inspect psro_runs/gen_NNN/ppo/checkpoints/ -->

  python run_psro.py eval-meta-game  --save-dir psro_runs --n-games 500
  python run_psro.py add-agent       --save-dir psro_runs \\
      --ppo-ckpt psro_runs/gen_001/ppo/checkpoints/ckpt_update_000300.pt

  python run_psro.py status          --save-dir psro_runs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from policy_pool import PolicyPool, RandomAgent


# ──────────────────────────────────────────────────────────────────────────────
# Pool state  (persistence layer)
# ──────────────────────────────────────────────────────────────────────────────

_STATE_FILE = "pool_state.json"


def _state_path(save_dir: str) -> str:
    return os.path.join(save_dir, _STATE_FILE)


def _load_pool(save_dir: str, device: str = "cpu") -> tuple[PolicyPool, dict]:
    """
    Reconstruct a PolicyPool from pool_state.json and return it alongside
    the raw state dict (for inspection / re-saving).

    Also restores the AlphaRank distribution if one was previously saved,
    so opponent sampling is immediately weighted correctly.

    Raises FileNotFoundError if the state file does not exist — run
    `init` first.
    """
    path = _state_path(save_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pool state not found at '{path}'.\n"
            "Run  python run_psro.py init --save-dir <dir>  first."
        )

    with open(path) as f:
        state = json.load(f)

    # Reconstruct pool without calling __init__ (which auto-inserts a RandomAgent)
    pool = PolicyPool.__new__(PolicyPool)
    pool.population   = []
    pool.distribution = None   # set below if present in state

    for entry in state["agents"]:
        if entry["type"] == "RandomAgent":
            pool.population.append(RandomAgent())
        elif entry["type"] == "PPOAgent":
            pool.add_policy(
                ppo_path    = entry["ppo_path"],
                belief_path = entry["belief_path"],
                device      = device,
            )
        else:
            raise ValueError(f"Unknown agent type in pool state: {entry['type']}")

    # Restore AlphaRank distribution if available
    if "alpharank_distribution" in state:
        import numpy as np
        dist = np.array(state["alpharank_distribution"])
        if len(dist) == len(pool):
            pool.set_distribution(dist)
        else:
            print(
                f"[PSRO] WARNING: Saved distribution length {len(dist)} "
                f"!= pool size {len(pool)} — ignoring stale distribution."
            )

    return pool, state


def _save_pool_state(pool: PolicyPool, state: dict, save_dir: str) -> None:
    """Serialise the current pool back to pool_state.json."""
    state["pool_size"]  = len(pool)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Persist AlphaRank distribution if one has been set
    if pool.distribution is not None:
        state["alpharank_distribution"] = pool.distribution.tolist()
    else:
        # Remove stale distribution if pool was reset to uniform
        state.pop("alpharank_distribution", None)

    path = _state_path(save_dir)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[PSRO] Pool state saved → {path}")


def _append_agent_entry(
    state:       dict,
    ppo_path:    str,
    belief_path: str,
) -> None:
    """Append a new PPOAgent descriptor to the in-memory state dict."""
    state.setdefault("agents", []).append({
        "type":        "PPOAgent",
        "ppo_path":    ppo_path,
        "belief_path": belief_path,
    })


def _current_gen(state: dict) -> int:
    return state.get("generation", 0)


def _belief_ckpt(save_dir: str) -> str:
    return os.path.join(save_dir, "belief", "belief_final.pt")


def _gen_dir(save_dir: str, gen: int) -> str:
    return os.path.join(save_dir, f"gen_{gen:03d}")


def _append_manifest(
    save_dir:    str,
    gen:         int,
    pool:        PolicyPool,
    ppo_path:    str,
    belief_path: str,
) -> None:
    path  = os.path.join(save_dir, "pool_manifest.jsonl")
    entry = {
        "generation":  gen,
        "pool_size":   len(pool),
        "ppo_added":   ppo_path,
        "belief_used": belief_path,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[PSRO] Manifest updated → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand implementations
# ──────────────────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    """Create a fresh pool state seeded with a single RandomAgent."""
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    path = _state_path(save_dir)
    if os.path.exists(path) and not getattr(args, "force", False):
        print(
            f"[PSRO] Pool state already exists at '{path}'.\n"
            "       Pass --force to overwrite."
        )
        return

    state = {
        "generation": 0,
        "pool_size":  1,
        "agents": [{"type": "RandomAgent"}],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"[PSRO] Initialised pool (gen 0) → {path}")
    print("       Pool: [RandomAgent × 1]")
    print("\n  Next step:")
    print(f"    python run_psro.py train-belief --save-dir {save_dir}")


def cmd_status(args: argparse.Namespace) -> None:
    """Print current pool contents, checkpoint locations, and AlphaRank weights."""
    pool, state = _load_pool(args.save_dir)
    gen         = _current_gen(state)
    ckpt        = _belief_ckpt(args.save_dir)
    ckpt_exists = "✓" if os.path.exists(ckpt) else "✗ (not yet trained)"

    print(f"\n{'─'*60}")
    print(f"  PSRO Status   save-dir: {args.save_dir}")
    print(f"{'─'*60}")
    print(f"  Generation   : {gen}")
    print(f"  Pool         : {pool}")
    print(f"  Belief ckpt  : {ckpt}  {ckpt_exists}")
    print(f"{'─'*60}")
    for i, entry in enumerate(state["agents"]):
        if entry["type"] == "RandomAgent":
            print(f"  [{i}] RandomAgent")
        else:
            ppo_ok    = "✓" if os.path.exists(entry["ppo_path"])    else "✗ missing"
            belief_ok = "✓" if os.path.exists(entry["belief_path"]) else "✗ missing"
            print(f"  [{i}] PPOAgent")
            print(f"       ppo    : {entry['ppo_path']}  {ppo_ok}")
            print(f"       belief : {entry['belief_path']}  {belief_ok}")

    # Show AlphaRank distribution if available
    if pool.distribution is not None:
        sampling_label = "AlphaRank"
        print(f"\n  AlphaRank distribution  ({sampling_label} sampling)")
        for i, (agent, w) in enumerate(zip(pool.population, pool.distribution)):
            bar = "█" * max(1, int(w * 30))
            print(f"    [{i}] {type(agent).__name__:<12}  {w:.3f}  {bar}")
    else:
        print(f"\n  Sampling: uniform (run eval-meta-game to compute AlphaRank weights)")

    print(f"{'─'*60}")

    # Suggest next step
    next_gen    = gen + 1
    gen_ppo_dir = os.path.join(_gen_dir(args.save_dir, next_gen), "ppo", "checkpoints")
    if os.path.isdir(gen_ppo_dir):
        print(f"\n  Pending PPO run: {gen_ppo_dir}")
        print("  → Run eval-meta-game, then add-agent.")
    else:
        print(f"\n  → Next: train-belief (or eval-meta-game if pool recently grew)")
    print()


def cmd_train_belief(args: argparse.Namespace) -> None:
    """Train / resume the belief network against the current pool."""
    from train_belief_net import train as train_belief

    pool, state = _load_pool(args.save_dir, device=args.device or "cpu")
    gen         = _current_gen(state)
    ckpt        = _belief_ckpt(args.save_dir)
    belief_dir  = os.path.dirname(ckpt)

    print(f"\n[PSRO] train-belief  |  gen={gen}  |  pool={pool}")
    if os.path.exists(ckpt):
        print(f"[PSRO] Resuming from {ckpt}")
    else:
        print("[PSRO] No prior belief checkpoint — training from scratch.")

    train_belief(
        policy_pool      = pool,
        resume_from      = ckpt if os.path.exists(ckpt) else None,
        num_games        = args.belief_games,
        games_per_loop   = args.belief_games_per_loop,
        updates_per_loop = args.belief_updates_per_loop,
        batch_size       = args.belief_batch_size,
        lr               = args.belief_lr,
        buffer_size      = args.belief_buffer_size,
        buffer_warmup    = args.belief_buffer_warmup,
        save_every       = args.belief_save_every,
        log_every        = args.belief_log_every,
        save_dir         = belief_dir,
        device_str       = args.device,
    )

    print(f"\n[PSRO] Belief network updated → {ckpt}")
    print("\n  Next step:")
    print(f"    python run_psro.py train-ppo --save-dir {args.save_dir}")


def cmd_train_ppo(args: argparse.Namespace) -> None:
    """
    Train a PPO best response against the current pool.

    Periodic checkpoints are saved to gen_NNN/ppo/checkpoints/ every 50
    updates so you can evaluate multiple snapshots from the same run before
    choosing which one to add.
    """
    from train_ppo import train_ppo

    pool, state = _load_pool(args.save_dir, device=args.device or "cpu")
    gen         = _current_gen(state) + 1     # this will become gen N+1
    ckpt        = _belief_ckpt(args.save_dir)
    ppo_dir     = os.path.join(_gen_dir(args.save_dir, gen), "ppo")

    if not os.path.exists(ckpt):
        print(
            f"[PSRO] WARNING: No belief checkpoint at '{ckpt}'.\n"
            "       Channels 7–9 will use random weights.  "
            "Run train-belief first for best results."
        )

    print(f"\n[PSRO] train-ppo  |  gen={gen}  |  pool={pool}")
    print(f"[PSRO] Belief checkpoint : {ckpt}")
    print(f"[PSRO] Output directory  : {ppo_dir}")
    print(f"[PSRO] Periodic checkpts : {ppo_dir}/checkpoints/  (every 50 updates)")

    ppo_final = train_ppo(
        policy_pool        = pool,
        belief_model_path  = ckpt,
        total_timesteps    = args.ppo_timesteps,
        save_dir           = ppo_dir,
        num_envs           = args.ppo_num_envs,
        num_steps          = args.ppo_num_steps,
        learning_rate      = args.ppo_lr,
        minibatch_size     = args.ppo_minibatch,
        update_epochs      = args.ppo_epochs,
        device_str         = args.device,
        run_name           = f"psro_gen{gen:03d}",
    )

    print(f"\n[PSRO] PPO training complete.")
    print(f"  agent_final.pt  : {ppo_final}")
    print(f"  Periodic ckpts  : {ppo_dir}/checkpoints/")
    print("\n  Evaluate the checkpoints, then commit your chosen one:")
    print(
        f"\n    # (Optional but recommended) Run meta-game eval to update AlphaRank weights:"
        f"\n    python run_psro.py eval-meta-game --save-dir {args.save_dir}"
        f"\n\n    # Commit chosen checkpoint:"
        f"\n    python run_psro.py add-agent --save-dir {args.save_dir} \\\n"
        f"        --ppo-ckpt <path/to/chosen_checkpoint.pt>\n"
    )


def cmd_add_agent(args: argparse.Namespace) -> None:
    """
    Register a manually chosen PPO checkpoint into the pool and advance
    the generation counter.

    The belief checkpoint defaults to the current belief_final.pt, which is
    the frozen belief model the PPOAgent will carry into future episodes.
    Pass --belief-ckpt to override (e.g. to lock in an earlier snapshot).

    The AlphaRank distribution is automatically cleared because it was
    computed over the OLD pool.  Run eval-meta-game after add-agent to
    recompute it for the updated pool before the next training round.
    """
    pool, state = _load_pool(args.save_dir, device="cpu")

    ppo_path     = args.ppo_ckpt
    belief_src   = args.belief_ckpt or _belief_ckpt(args.save_dir)
    new_gen      = _current_gen(state) + 1
    # belief_path = args.belief_ckpt or _belief_ckpt(args.save_dir)

    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO checkpoint not found: '{ppo_path}'")
    if not os.path.exists(belief_src):
        raise FileNotFoundError(f"Belief checkpoint not found: '{belief_src}'")

    frozen_belief_dir  = os.path.join(_gen_dir(args.save_dir, new_gen), "belief")
    os.makedirs(frozen_belief_dir, exist_ok=True)
    frozen_belief_path = os.path.join(frozen_belief_dir, "belief_frozen.pt")
    shutil.copy2(belief_src, frozen_belief_path)
    print(
        f"[PSRO] Belief snapshot: {belief_src}\n"
        f"       → {frozen_belief_path}  (immutable frozen copy)"
    )
    state["generation"] = new_gen

    _append_agent_entry(state, ppo_path, frozen_belief_path)
    pool.add_policy(ppo_path=ppo_path, belief_path=frozen_belief_path, device="cpu")

    # The old distribution is now stale — clear it so training falls back to
    # uniform until eval-meta-game recomputes the AlphaRank weights.
    pool.clear_distribution()
    state.pop("alpharank_distribution", None)

    _save_pool_state(pool, state, args.save_dir)
    _append_manifest(args.save_dir, new_gen, pool, ppo_path, frozen_belief_path)

    print(f"\n[PSRO] Generation {new_gen} committed.")
    print(f"  Pool : {pool}")
    print(
        f"\n  AlphaRank distribution cleared (pool grew).\n"
        f"  Run eval-meta-game to recompute before the next training round:\n\n"
        f"    python run_psro.py eval-meta-game --save-dir {args.save_dir}\n"
        f"\n  Or skip eval and proceed with uniform sampling:\n"
        f"    python run_psro.py train-belief --save-dir {args.save_dir}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# eval-meta-game
# ──────────────────────────────────────────────────────────────────────────────

def cmd_eval_meta_game(args: argparse.Namespace) -> None:
    """
    Evaluate the current pool's meta-game and compute the AlphaRank
    meta-Nash distribution.

    What it does
    ────────────
    1. Generates all unique 4-agent combos from the pool (with replacement),
       exploiting GongZhu's symmetry to minimise the number of matchups.
    2. Simulates --n-games games per matchup to estimate average payoffs.
    3. Fills the N×N×N×N payoff tensor (backfilled by permutation symmetry).
    4. Runs OpenSpiel's AlphaRank solver to obtain the meta-Nash distribution.
    5. Saves the tensor (.npy) and distribution (.json) for inspection.
    6. Writes the distribution back to pool_state.json so that subsequent
       train-belief and train-ppo calls sample from it automatically.

    Run this AFTER add-agent (because adding a new agent invalidates the
    previous distribution) and BEFORE the next train-belief / train-ppo.
    You can skip it and train with uniform sampling — AlphaRank is optional.

    OpenSpiel
    ─────────
    If OpenSpiel is not installed, the command still runs but returns a
    uniform distribution.  Install with:  pip install open_spiel
    """
    from eval_meta_game import evaluate as eval_meta

    pool, state = _load_pool(args.save_dir, device="cpu")
    gen         = _current_gen(state)

    print(f"\n[PSRO] eval-meta-game  |  gen={gen}  |  pool={pool}")

    eval_dir = os.path.join(args.save_dir, f"eval_gen_{gen:03d}")
    distribution = eval_meta(
        pool        = pool,
        save_dir    = eval_dir,
        n_games     = args.n_games,
        alpha       = args.alpha,
        master_seed = args.seed,
        verbose     = True,
    )

    # Store distribution on pool and persist to pool_state.json
    pool.set_distribution(distribution)
    state["alpharank_distribution"] = distribution.tolist()
    _save_pool_state(pool, state, args.save_dir)

    print(
        f"[PSRO] AlphaRank distribution stored in pool_state.json.\n"
        f"       Subsequent train-belief and train-ppo calls will sample "
        f"opponents using these weights.\n"
        f"\n  Next step:"
        f"\n    python run_psro.py train-belief --save-dir {args.save_dir}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# run-auto  (fully automated convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def cmd_run_auto(args: argparse.Namespace) -> None:
    """
    Fully automated PSRO loop — belief → PPO → (eval) → add-agent, N times.
    Always picks agent_final.pt as the committed checkpoint.

    Pass --skip-eval to disable the AlphaRank evaluation step and use uniform
    opponent sampling throughout (faster, but no meta-Nash weighting).

    For manual checkpoint selection, use the individual subcommands instead.
    """
    from train_belief_net import train as train_belief
    from train_ppo import train_ppo

    # Auto-init if the state file doesn't exist yet
    if not os.path.exists(_state_path(args.save_dir)):
        cmd_init(args)

    loop_start = time.time()

    for _ in range(args.num_generations):
        pool, state = _load_pool(args.save_dir, device=args.device or "cpu")
        gen         = _current_gen(state) + 1
        ckpt        = _belief_ckpt(args.save_dir)
        belief_dir  = os.path.dirname(ckpt)
        ppo_dir     = os.path.join(_gen_dir(args.save_dir, gen), "ppo")

        sampling_mode = "AlphaRank" if pool.distribution is not None else "uniform"
        print(f"\n{'═'*64}")
        print(f"  PSRO  Generation {gen}   Pool: {pool}   Sampling: {sampling_mode}")
        print(f"{'═'*64}")

        # Step 1: belief
        print(f"\n[PSRO Gen {gen}] Belief training ({args.belief_games} games)…")
        train_belief(
            policy_pool      = pool,
            resume_from      = ckpt if os.path.exists(ckpt) else None,
            num_games        = args.belief_games,
            games_per_loop   = args.belief_games_per_loop,
            updates_per_loop = args.belief_updates_per_loop,
            batch_size       = args.belief_batch_size,
            lr               = args.belief_lr,
            buffer_size      = args.belief_buffer_size,
            buffer_warmup    = args.belief_buffer_warmup,
            save_every       = args.belief_save_every,
            log_every        = args.belief_log_every,
            save_dir         = belief_dir,
            device_str       = args.device,
        )

        # Step 2: PPO best response
        print(f"\n[PSRO Gen {gen}] PPO training ({args.ppo_timesteps:,} steps)…")
        ppo_final = train_ppo(
            policy_pool        = pool,
            belief_model_path  = ckpt,
            total_timesteps    = args.ppo_timesteps,
            save_dir           = ppo_dir,
            num_envs           = args.ppo_num_envs,
            num_steps          = args.ppo_num_steps,
            learning_rate      = args.ppo_lr,
            minibatch_size     = args.ppo_minibatch,
            update_epochs      = args.ppo_epochs,
            device_str         = args.device,
            run_name           = f"psro_gen{gen:03d}",
        )

        # Step 3: commit agent_final.pt
        # Clear stale distribution first (pool is about to grow)
        frozen_belief_dir  = os.path.join(_gen_dir(args.save_dir, gen), "belief")
        os.makedirs(frozen_belief_dir, exist_ok=True)
        frozen_belief_path = os.path.join(frozen_belief_dir, "belief_frozen.pt")
        shutil.copy2(ckpt, frozen_belief_path)
        print(f"[PSRO Gen {gen}] Belief snapshot → {frozen_belief_path}")

        pool.clear_distribution()
        state.pop("alpharank_distribution", None)
        state["generation"] = gen
        _append_agent_entry(state, ppo_final, frozen_belief_path)
        pool.add_policy(ppo_path=ppo_final, belief_path=frozen_belief_path, device="cpu")
        _save_pool_state(pool, state, args.save_dir)
        _append_manifest(args.save_dir, gen, pool, ppo_final, frozen_belief_path)

        # Step 4 (optional): AlphaRank evaluation to update sampling weights
        if not args.skip_eval:
            from eval_meta_game import evaluate as eval_meta
            print(f"\n[PSRO Gen {gen}] Meta-game evaluation ({args.eval_n_games} games/matchup)…")
            eval_dir     = os.path.join(args.save_dir, f"eval_gen_{gen:03d}")
            distribution = eval_meta(
                pool        = pool,
                save_dir    = eval_dir,
                n_games     = args.eval_n_games,
                alpha       = args.eval_alpha,
                verbose     = True,
            )
            pool.set_distribution(distribution)
            state["alpharank_distribution"] = distribution.tolist()
            _save_pool_state(pool, state, args.save_dir)

        elapsed = time.time() - loop_start
        print(f"\n[PSRO Gen {gen}] Complete.  Total elapsed: {elapsed/60:.1f} min")

    pool, _ = _load_pool(args.save_dir)
    print(f"\n{'═'*64}")
    print(f"  run-auto complete.  Final pool: {pool}")
    print(f"  Total time: {(time.time() - loop_start)/60:.1f} min")
    print(f"{'═'*64}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save-dir", type=str, default="psro_runs",
                   help="Root directory for all checkpoints and logs.")
    p.add_argument("--device",   type=str, default=None,
                   help="Torch device ('cuda', 'cpu', or None for auto-detect).")


def _add_belief_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("belief-net hyperparameters")
    g.add_argument("--belief-games",            type=int,   default=50_000)
    g.add_argument("--belief-games-per-loop",   type=int,   default=10)
    g.add_argument("--belief-updates-per-loop", type=int,   default=10)
    g.add_argument("--belief-batch-size",       type=int,   default=256)
    g.add_argument("--belief-lr",               type=float, default=5e-4)
    g.add_argument("--belief-buffer-size",      type=int,   default=100_000)
    g.add_argument("--belief-buffer-warmup",    type=int,   default=10_000)
    g.add_argument("--belief-save-every",       type=int,   default=5_000)
    g.add_argument("--belief-log-every",        type=int,   default=500)


def _add_ppo_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("ppo hyperparameters")
    g.add_argument("--ppo-timesteps", type=int,   default=5_000_000)
    g.add_argument("--ppo-num-envs",  type=int,   default=16)
    g.add_argument("--ppo-num-steps", type=int,   default=260)
    g.add_argument("--ppo-lr",        type=float, default=2.5e-4)
    g.add_argument("--ppo-minibatch", type=int,   default=520)
    g.add_argument("--ppo-epochs",    type=int,   default=4)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog        = "run_psro.py",
        description = "PSRO orchestrator for GongZhu — manual or automated.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
examples:
  # One-time setup
  python run_psro.py init --save-dir psro_runs

  # Manual generation cycle (with AlphaRank)
  python run_psro.py train-belief   --save-dir psro_runs
  python run_psro.py train-ppo      --save-dir psro_runs
  python run_psro.py eval-meta-game --save-dir psro_runs --n-games 500
  python run_psro.py add-agent      --save-dir psro_runs \\
      --ppo-ckpt psro_runs/gen_001/ppo/checkpoints/ckpt_update_000300.pt

  # Check pool state and AlphaRank weights at any time
  python run_psro.py status --save-dir psro_runs

  # Fully automated with AlphaRank eval (default)
  python run_psro.py run-auto --save-dir psro_runs --num-generations 5

  # Fully automated without AlphaRank (faster, uniform sampling)
  python run_psro.py run-auto --save-dir psro_runs --num-generations 5 --skip-eval
        """,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init",
        help="Create a fresh pool seeded with RandomAgent.")
    _add_shared_args(p_init)
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite an existing pool_state.json.")

    # status
    p_status = sub.add_parser("status",
        help="Print current pool contents and checkpoint paths.")
    _add_shared_args(p_status)

    # train-belief
    p_tb = sub.add_parser("train-belief",
        help="Train / resume the belief network against the current pool.")
    _add_shared_args(p_tb)
    _add_belief_args(p_tb)

    # train-ppo
    p_tp = sub.add_parser("train-ppo",
        help="Train a PPO best response; saves periodic checkpoints for review.")
    _add_shared_args(p_tp)
    _add_ppo_args(p_tp)

    # eval-meta-game
    p_eval = sub.add_parser("eval-meta-game",
        help="Simulate matchups, build payoff tensor, run AlphaRank.")
    _add_shared_args(p_eval)
    p_eval.add_argument("--n-games", type=int,   default=1_000,
                        help="Games simulated per unique 4-agent matchup.")
    p_eval.add_argument("--alpha",   type=float, default=1e-2,
                        help="AlphaRank temperature (higher = more selective).")
    p_eval.add_argument("--seed",    type=int,   default=42)

    # add-agent
    p_add = sub.add_parser("add-agent",
        help="Register a chosen PPO checkpoint into the pool after manual eval.")
    _add_shared_args(p_add)
    p_add.add_argument(
        "--ppo-ckpt", type=str, required=True,
        help="Path to the PPO checkpoint you have chosen to add to the pool.",
    )
    p_add.add_argument(
        "--belief-ckpt", type=str, default=None,
        help=(
            "Belief checkpoint to freeze into this agent.  "
            "Defaults to <save-dir>/belief/belief_final.pt."
        ),
    )

    # run-auto
    p_auto = sub.add_parser("run-auto",
        help="Fully automated loop for N generations (always picks agent_final.pt).")
    _add_shared_args(p_auto)
    _add_belief_args(p_auto)
    _add_ppo_args(p_auto)
    p_auto.add_argument("--num-generations", type=int, default=10,
                        help="Number of PSRO generations to run.")
    # AlphaRank eval options for run-auto
    eval_group = p_auto.add_argument_group(
        "AlphaRank evaluation (runs after each add-agent step)"
    )
    eval_group.add_argument("--skip-eval",   action="store_true",
                            help="Skip AlphaRank evaluation; use uniform sampling throughout.")
    eval_group.add_argument("--eval-n-games", type=int,   default=200,
                            help="Games per matchup for the automated eval step.")
    eval_group.add_argument("--eval-alpha",   type=float, default=1e-2,
                            help="AlphaRank temperature for the automated eval step.")

    return root


def main() -> None:
    dispatch = {
        "init":            cmd_init,
        "status":          cmd_status,
        "train-belief":    cmd_train_belief,
        "train-ppo":       cmd_train_ppo,
        "eval-meta-game":  cmd_eval_meta_game,
        "add-agent":       cmd_add_agent,
        "run-auto":        cmd_run_auto,
    }
    args = build_parser().parse_args()
    dispatch[args.command](args)


if __name__ == "__main__":
    main()