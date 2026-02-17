"""
ppo_sft_sb3.py — PPO + Supervised Fine-Tuning for Gridlock (and beyond)
using Stable-Baselines3 + sb3-contrib.

Dependencies:
    pip install stable-baselines3 sb3-contrib tensorboard

Phase 1 — SFT  : Warm-start the actor with cross-entropy on random expert
                  rollouts that achieved a minimum score threshold.
Phase 2 — PPO  : Fine-tune with MaskablePPO (invalid-action masking built in).

To adapt to a different environment:
  1. Change ENV_ID / make_env() to point at your new env.
  2. Tune the CONFIG block.
  3. If your env doesn't support action masking, swap MaskablePPO → PPO
     and drop the ActionMasker wrapper + mask_fn.
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from gridlock.sb3_gym_env import GridlockEnv, score_grid


# ===========================================================================
# CONFIG — edit this block to adapt to a new task
# ===========================================================================
CONFIG = dict(
    exp_name          = "Gridlock_PPO_SFT",
    seed              = 1,
    # --- PPO ---
    total_timesteps   = 5_000_000,
    learning_rate     = 2.5e-4,
    n_envs            = 8,
    n_steps           = 256,       # rollout steps per env per update
    batch_size        = 64,        # minibatch size
    n_epochs          = 4,         # PPO update epochs
    gamma             = 0.99,
    gae_lambda        = 0.95,
    clip_range        = 0.2,
    ent_coef          = 0.01,
    vf_coef           = 0.5,
    max_grad_norm     = 0.5,
    # --- Network ---
    hidden_size       = 128,       # units in each hidden layer
    # --- SFT ---
    do_sft            = True,
    sft_demos         = 5_000,     # target number of qualifying games
    sft_threshold     = 20,        # minimum score to count as "expert"
    sft_epochs        = 10,
    sft_lr            = 3e-4,
    sft_batch_size    = 256,
    # --- Eval / Early stopping ---
    eval_freq         = 10_000,    # env steps between evaluations
    eval_episodes     = 50,
    patience          = 50,        # eval intervals with no improvement → stop
)
# ===========================================================================


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def mask_fn(env: gym.Env) -> np.ndarray:
    """Passed to ActionMasker; retrieves the valid-action mask."""
    return env.action_masks()


def make_env(seed: int = 0):
    """
    Returns a single wrapped environment.
    Swap GridlockEnv for any other gym.Env subclass here.
    """
    def _init():
        env = GridlockEnv()
        env = ActionMasker(env, mask_fn)
        env = RecordEpisodeStatistics(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Supervised Fine-Tuning helpers
# ---------------------------------------------------------------------------

def collect_expert_data(
    min_score: float,
    num_games: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Play random valid moves and keep episodes whose final score ≥ min_score.
    Returns (observations, actions) arrays ready for supervised training.
    """
    print(f"Generating {num_games} expert demos (score ≥ {min_score})…")
    env = GridlockEnv()
    rng = np.random.RandomState(seed)
    obs_list, act_list = [], []
    collected, attempts = 0, 0
    max_attempts = num_games * 10

    while collected < num_games and attempts < max_attempts:
        obs, info = env.reset(seed=seed + attempts)
        attempts += 1
        done = False
        temp_obs, temp_act = [], []

        while not done:
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = rng.choice(valid)
            temp_obs.append(obs)
            temp_act.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done and reward >= min_score:
                obs_list.extend(temp_obs)
                act_list.extend(temp_act)
                collected += 1
                if collected % 500 == 0:
                    print(f"  {collected}/{num_games} collected ({attempts} attempts)")

    if collected < num_games:
        print(f"Warning: only {collected}/{num_games} demos after {attempts} attempts.")
    else:
        print(f"Collection complete — {attempts} attempts.")

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def run_sft(model: MaskablePPO, cfg: dict, writer: SummaryWriter, device: torch.device):
    """
    Supervised fine-tuning of the actor network inside an SB3 MaskablePPO model.

    SB3 policy internals used:
        policy.extract_features(obs)          → shared feature vector
        policy.mlp_extractor(features)        → (latent_pi, latent_vf)
        policy.action_net(latent_pi)          → raw action logits
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — SUPERVISED FINE-TUNING")
    print("=" * 70)

    expert_obs, expert_acts = collect_expert_data(
        min_score=cfg["sft_threshold"],
        num_games=cfg["sft_demos"],
        seed=cfg["seed"],
    )
    if len(expert_obs) == 0:
        print("ERROR: no expert demos collected — skipping SFT.")
        return

    dataset = TensorDataset(
        torch.from_numpy(expert_obs).to(device),
        torch.from_numpy(expert_acts).to(device),
    )
    loader = DataLoader(dataset, batch_size=cfg["sft_batch_size"], shuffle=True)

    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["sft_lr"], eps=1e-5)

    print(f"Training on {len(expert_obs)} transitions for {cfg['sft_epochs']} epochs…")
    for epoch in range(cfg["sft_epochs"]):
        total_loss, correct, total = 0.0, 0, 0

        for b_obs, b_act in loader:
            optimizer.zero_grad()

            # Forward pass through SB3 policy actor
            with torch.set_grad_enabled(True):
                features   = policy.extract_features(b_obs, policy.pi_features_extractor)
                latent_pi, _ = policy.mlp_extractor(features)
                logits     = policy.action_net(latent_pi)

            if torch.isnan(logits).any():
                print(f"  NaN in logits at epoch {epoch+1} — skipping batch.")
                continue

            loss = nn.functional.cross_entropy(logits, b_act)
            if torch.isnan(loss):
                print(f"  NaN loss at epoch {epoch+1} — skipping batch.")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == b_act).sum().item()
            total      += b_act.size(0)

        epoch_loss = total_loss / max(len(loader), 1)
        epoch_acc  = correct   / max(total, 1)
        print(f"  Epoch {epoch+1}/{cfg['sft_epochs']}: loss={epoch_loss:.4f}  acc={epoch_acc:.2%}")
        writer.add_scalar("sft/loss",     epoch_loss, epoch)
        writer.add_scalar("sft/accuracy", epoch_acc,  epoch)

    print("SFT complete.\n")

    # Quick sanity-check
    with torch.no_grad():
        dummy = torch.rand(8, 10, device=device) * 0.1 + 0.5
        feats = policy.extract_features(dummy, policy.pi_features_extractor)
        lpi, _ = policy.mlp_extractor(feats)
        logits = policy.action_net(lpi)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("Policy is producing NaN/Inf after SFT — aborting.")
        print(f"✓ Actor logit range after SFT: [{logits.min():.3f}, {logits.max():.3f}]")


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------

class SFTCallback(BaseCallback):
    """Runs SFT once before PPO training begins (step 0)."""

    def __init__(self, cfg, writer, device, verbose=0):
        super().__init__(verbose)
        self.cfg    = cfg
        self.writer = writer
        self.device = device
        self._done  = False

    def _on_training_start(self):
        if self.cfg["do_sft"] and not self._done:
            run_sft(self.model, self.cfg, self.writer, self.device)
            # Reset SB3's internal optimizer so SFT LR doesn't pollute PPO phase
            self.model.policy.optimizer = torch.optim.Adam(
                self.model.policy.parameters(),
                lr=self.cfg["learning_rate"],
                eps=1e-5,
            )
            self._done = True

    def _on_step(self) -> bool:
        return True


class TBExtraCallback(BaseCallback):
    """
    Logs a few extra scalars (SPS, mean rollout reward) to TensorBoard
    on top of what SB3 already writes.
    """

    def __init__(self, writer: SummaryWriter, log_freq: int = 2048, verbose=0):
        super().__init__(verbose)
        self.writer   = writer
        self.log_freq = log_freq
        self._start   = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            sps = int(self.num_timesteps / (time.time() - self._start))
            self.writer.add_scalar("charts/SPS", sps, self.num_timesteps)
        return True


class MaskableEvalCallback(EvalCallback):
    """
    Thin wrapper around EvalCallback that enables action masking during
    evaluation (the base class does not pass masks by default for MaskablePPO).
    We also log results to our own TensorBoard writer.
    """

    def __init__(self, *args, writer: SummaryWriter, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = writer

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.tb_writer.add_scalar(
                "eval/mean_reward", self.last_mean_reward, self.num_timesteps
            )
        return result


# ---------------------------------------------------------------------------
# Policy evaluation helper (kept for final detailed report)
# ---------------------------------------------------------------------------

def evaluate_gridlock(
    model: MaskablePPO,
    n_eval: int = 200,
    deterministic: bool = False,
) -> tuple[float, float, float]:
    """
    Evaluate a MaskablePPO model over n_eval Gridlock episodes.

    Returns (mean_score, success_rate, mean_episode_length).
    success_rate = fraction of episodes with score > 0 (at least one line).
    """
    env = GridlockEnv()
    scores, lengths = [], []

    for _ in range(n_eval):
        obs, info = env.reset()
        done = False
        steps = 0
        reward = 0.0

        while not done:
            mask = env.action_masks()
            # MaskablePPO.predict accepts action_masks kwarg
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1

        if reward > 0:
            scores.append(reward)
        lengths.append(steps)

    mean_score   = float(np.mean(scores))   if scores  else 0.0
    success_rate = len(scores) / n_eval
    mean_length  = float(np.mean(lengths))
    return mean_score, success_rate, mean_length


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = CONFIG

    # Reproducibility
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name  = f"{cfg['exp_name']}_{int(time.time())}"
    log_dir   = f"runs/{run_name}"
    model_dir = f"{log_dir}/models"
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in cfg.items()),
    )

    print(f"Device : {device}")
    print(f"Run    : {run_name}\n")

    # ------------------------------------------------------------------
    # Vectorised training environments
    # ------------------------------------------------------------------
    train_env = make_vec_env(
        make_env(seed=cfg["seed"]),
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
    )

    # Separate single-env for evaluation inside EvalCallback
    eval_env = make_vec_env(make_env(seed=cfg["seed"] + 999), n_envs=1)

    # ------------------------------------------------------------------
    # Policy network architecture
    # net_arch mirrors the CleanRL agent: two hidden layers of hidden_size
    # with Tanh activations, separate actor / critic streams.
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        net_arch        = dict(pi=[cfg["hidden_size"], cfg["hidden_size"]],
                               vf=[cfg["hidden_size"], cfg["hidden_size"]]),
        activation_fn   = nn.Tanh,
        ortho_init      = True,   # orthogonal weight initialisation (CleanRL default)
    )

    # ------------------------------------------------------------------
    # MaskablePPO model
    # ------------------------------------------------------------------
    model = MaskablePPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = cfg["learning_rate"],
        n_steps         = cfg["n_steps"],
        batch_size      = cfg["batch_size"],
        n_epochs        = cfg["n_epochs"],
        gamma           = cfg["gamma"],
        gae_lambda      = cfg["gae_lambda"],
        clip_range      = cfg["clip_range"],
        ent_coef        = cfg["ent_coef"],
        vf_coef         = cfg["vf_coef"],
        max_grad_norm   = cfg["max_grad_norm"],
        policy_kwargs   = policy_kwargs,
        tensorboard_log = log_dir,
        verbose         = 1,
        seed            = cfg["seed"],
        device          = device,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    no_improve_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals = cfg["patience"],
        verbose                  = 1,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = max(cfg["eval_freq"] // cfg["n_envs"], 1),
        n_eval_episodes      = cfg["eval_episodes"],
        deterministic        = False,
        callback_after_eval  = no_improve_cb,
        writer               = writer,
        verbose              = 1,
    )

    sft_cb  = SFTCallback(cfg, writer, device)
    tb_cb   = TBExtraCallback(writer, log_freq=cfg["n_steps"] * cfg["n_envs"])

    # ------------------------------------------------------------------
    # Train  (SFT fires automatically at the start via SFTCallback)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 2 — PPO TRAINING")
    print("=" * 70 + "\n")

    model.learn(
        total_timesteps = cfg["total_timesteps"],
        callback        = [sft_cb, eval_cb, tb_cb],
        progress_bar    = True,
    )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    best_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_path):
        print(f"Loading best model from {best_path}")
        model = MaskablePPO.load(best_path, env=train_env, device=device)

    for deterministic, label in [(False, "Sampling"), (True, "Greedy/Argmax")]:
        score, success, length = evaluate_gridlock(model, n_eval=200, deterministic=deterministic)
        print(f"\nFinal ({label}):")
        print(f"  Mean Score     : {score:.2f}")
        print(f"  Success Rate   : {success:.1%}")
        print(f"  Mean Ep Length : {length:.1f}")
        writer.add_scalar(f"eval/final_score_{label.lower()}", score, cfg["total_timesteps"])

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    writer.close()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"TensorBoard logs : {log_dir}")
    print(f"{'='*70}\n")